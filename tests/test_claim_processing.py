"""
Full test suite for Claim Processing.

Run with:
    pytest tests/test_claim_processing.py -v

No live Ollama or GPU required — all LLM/model calls are mocked.
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
from unittest.mock import patch, MagicMock

from src.claim_processing.decomposer import ClaimDecomposer
from src.claim_processing.stance_classifier import StanceClassifier, StanceLabel, StanceResult
from src.claim_processing.text_cleaner import (
    clean_passage_text,
    clean_passages_in_retrieval_results,
)


class MockPassage:
    def __init__(self, text, source="mock_source", dataset="mock_dataset", id="mock_1"):
        self.text = text
        self.source = source
        self.dataset = dataset
        self.id = id


class MockRetrievalResult:
    def __init__(self, score, rank, passage):
        self.score = score
        self.rank = rank
        self.passage = passage


class TestTextCleaner:
    """Tests for clean_passage_text and clean_passages_in_retrieval_results."""

    def test_strips_tab_annotations(self):
        """Core FEVER case: everything after the first tab must be removed."""
        raw = (
            "Henry was on the verge of ruin .\t"
            "Francis I of France\tFrancis I of France\tHoly Roman Emperor Charles V"
        )
        assert clean_passage_text(raw) == "Henry was on the verge of ruin."

    def test_strips_space_before_period(self):
        """FEVER sentences often end with ' .' after tab-stripping — must fix."""
        raw = "He had a recurring role on Fox .\tFox Broadcasting\tFox Broadcasting"
        assert clean_passage_text(raw) == "He had a recurring role on Fox."

    def test_clean_sentence_unchanged(self):
        """Sentences with no tabs and correct punctuation should pass through."""
        clean = "A clean sentence with no tabs."
        assert clean_passage_text(clean) == clean

    def test_empty_string_returns_empty(self):
        assert clean_passage_text("") == ""

    def test_adds_missing_terminal_period(self):
        """Sentences without terminal punctuation should get one added."""
        assert clean_passage_text("He was born in 1970") == "He was born in 1970."

    def test_preserves_exclamation_and_question(self):
        """Existing ! and ? must not be replaced with a period."""
        assert clean_passage_text("Is this a claim?") == "Is this a claim?"
        assert clean_passage_text("This is surprising!") == "This is surprising!"

    def test_clean_passages_in_retrieval_results_mutates_in_place(self):
        """The list helper must clean each r.passage.text and return the same list."""
        r1 = MockRetrievalResult(
            score=0.9, rank=1,
            passage=MockPassage(
                text="He had a role on Fox .\tFox Broadcasting\tFox Broadcasting",
            ),
        )
        r2 = MockRetrievalResult(
            score=0.8, rank=2,
            passage=MockPassage(text="A clean sentence."),
        )
        results = [r1, r2]
        returned = clean_passages_in_retrieval_results(results)

        assert returned is results  # same list object
        assert r1.passage.text == "He had a role on Fox."
        assert r2.passage.text == "A clean sentence."



class TestClaimDecomposer:

    def test_trivially_atomic_claim(self):
        """Short claims without conjunctions bypass the LLM."""
        decomposer = ClaimDecomposer()
        claim = "Lionel Messi plays for Inter Miami."

        result = decomposer.decompose(claim)

        assert result.was_compound is False
        assert len(result.atomic_claims) == 1
        assert result.atomic_claims[0].text == claim
        assert result.model_used == "passthrough"

    @patch("ollama.Client.chat")
    def test_compound_claim_decomposition(self, mock_chat):
        """Compound claims are properly parsed from the mocked LLM JSON response."""
        mock_chat.return_value = {
            "message": {
                "content": (
                    '{"atomic_claims": ["Messi won the World Cup.", '
                    '"Messi is from Argentina."], "reasoning": "Split by facts."}'
                )
            }
        }

        decomposer = ClaimDecomposer()
        claim = "Messi won the World Cup and he is from Argentina."

        result = decomposer.decompose(claim)

        assert result.was_compound is True
        assert len(result.atomic_claims) == 2
        assert result.atomic_claims[0].text == "Messi won the World Cup."
        assert result.atomic_claims[1].text == "Messi is from Argentina."
        assert mock_chat.call_count == 1

    def test_decomposer_empty_claim(self):
        """Empty / whitespace-only input is handled gracefully."""
        decomposer = ClaimDecomposer()
        result = decomposer.decompose("   ")

        assert len(result.atomic_claims) == 1
        assert result.error == "Empty claim text."

    @patch("ollama.Client.chat")
    def test_decomposer_post_processing_filters(self, mock_chat):
        """First-person pronouns are dropped and duplicates are removed."""
        mock_chat.return_value = {
            "message": {
                "content": (
                    '{"atomic_claims":["I think Messi is great.", '
                    '"Messi is from Argentina.", "Messi is from Argentina."], '
                    '"reasoning": "Split."}'
                )
            }
        }

        decomposer = ClaimDecomposer()
        result = decomposer.decompose("I think Messi is great and he is from Argentina.")

        assert len(result.atomic_claims) == 1
        assert result.atomic_claims[0].text == "Messi is from Argentina."

    @patch("ollama.Client.chat")
    def test_decomposer_ollama_failure_fallback(self, mock_chat):
        """If Ollama raises, the decomposer falls back to passthrough."""
        mock_chat.side_effect = Exception("Connection refused to Ollama server")

        decomposer = ClaimDecomposer(max_retries=0)
        claim = "Messi won the World Cup and he is from Argentina."

        result = decomposer.decompose(claim)

        assert result.model_used == "passthrough"
        assert len(result.atomic_claims) == 1
        assert result.atomic_claims[0].text == claim
        assert "Unexpected error" in result.error

    @patch("ollama.Client.chat")
    def test_decomposer_invalid_json_fallback(self, mock_chat):
        """If Ollama returns JSON that violates the Pydantic schema, fall back."""
        mock_chat.return_value = {
            "message": {"content": '{"wrong_key": "not a list"}'}
        }

        decomposer = ClaimDecomposer(max_retries=0)
        claim = "Messi won the World Cup and he is from Argentina."

        result = decomposer.decompose(claim)

        assert result.model_used == "passthrough"
        assert "Parse/validation error" in result.error

    @patch("ollama.Client.chat")
    def test_decomposer_model_not_found_does_not_retry(self, mock_chat):
        """'model not found' is non-recoverable — must not retry."""
        import ollama as _ollama
        mock_chat.side_effect = _ollama.ResponseError("model 'gemma3:4b' not found")

        decomposer = ClaimDecomposer(max_retries=3)
        result = decomposer.decompose("Messi won the World Cup and is from Argentina.")

        # Should have attempted exactly once, then given up
        assert mock_chat.call_count == 1
        assert result.model_used == "passthrough"

    @patch("ollama.Client.chat")
    def test_decompose_batch_returns_one_result_per_claim(self, mock_chat):
        """decompose_batch must return results in the same order as inputs."""
        mock_chat.return_value = {
            "message": {
                "content": '{"atomic_claims": ["Claim A."], "reasoning": "Already atomic."}'
            }
        }

        decomposer = ClaimDecomposer()
        claims = [
            "Messi won the World Cup and is from Argentina.",
            "Messi won the World Cup and is from Argentina.",
            "Messi won the World Cup and is from Argentina.",
        ]
        results = decomposer.decompose_batch(claims)

        assert len(results) == 3
        for r in results:
            assert len(r.atomic_claims) >= 1


class TestStanceClassifier:

    @patch.object(StanceClassifier, "_batch_infer")
    def test_stance_classifier_aggregation(self, mock_batch_infer):
        """Classifier correctly aggregates multiple supporting passages."""
        mock_batch_infer.return_value = [
            {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
            {"entailment": 0.8, "neutral": 0.1,  "contradiction": 0.1},
        ]

        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu")
            classifier._label_order = ["contradiction", "entailment", "neutral"]

        claim = "The sky is blue."
        retrievals = [
            MockRetrievalResult(0.9, 1, MockPassage("The sky appears blue.")),
            MockRetrievalResult(0.8, 2, MockPassage("Rayleigh scattering makes the sky blue.")),
        ]

        result = classifier.classify(claim, retrievals)

        assert isinstance(result, StanceResult)
        assert result.aggregate_label == StanceLabel.SUPPORTING
        assert result.supporting_count == 2
        assert result.refuting_count == 0

    def test_stance_classifier_empty_retrievals(self):
        """Empty retrieval list must return NEUTRAL with zero passage stances."""
        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu")

        result = classifier.classify("Some random claim.", [])

        assert result.aggregate_label == StanceLabel.NEUTRAL
        assert len(result.passage_stances) == 0
        assert result.aggregate_score == 0.0

    @patch.object(StanceClassifier, "_batch_infer")
    def test_stance_classifier_confidence_threshold_downgrade(self, mock_batch_infer):
        """Predictions below threshold must be downgraded to NEUTRAL."""
        mock_batch_infer.return_value = [
            {"entailment": 0.45, "neutral": 0.35, "contradiction": 0.20}
        ]

        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu", confidence_threshold=0.50)
            classifier._label_order = ["contradiction", "entailment", "neutral"]

        retrievals = [MockRetrievalResult(0.9, 1, MockPassage("Vague evidence."))]
        result = classifier.classify("Claim.", retrievals)

        assert result.passage_stances[0].stance == StanceLabel.NEUTRAL
        assert result.aggregate_label == StanceLabel.NEUTRAL

    @patch.object(StanceClassifier, "_batch_infer")
    def test_stance_classifier_conflicting_evidence_aggregation(self, mock_batch_infer):
        """Higher confidence * retrieval_score wins when evidence conflicts."""
        mock_batch_infer.return_value = [
            {"entailment": 0.9,  "neutral": 0.05, "contradiction": 0.05},
            {"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95},
        ]

        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu")
            classifier._label_order = ["contradiction", "entailment", "neutral"]

        retrievals = [
            MockRetrievalResult(0.4, 1, MockPassage("Weakly relevant supporting info.")),
            MockRetrievalResult(0.9, 2, MockPassage("Highly relevant refuting info.")),
        ]

        result = classifier.classify("Claim.", retrievals)

        assert result.aggregate_label == StanceLabel.REFUTING
        assert result.refuting_count == 1
        assert result.supporting_count == 1

    @patch.object(StanceClassifier, "_batch_infer")
    def test_classify_strips_fever_tab_annotations(self, mock_batch_infer):
        """Passage text must be cleaned before inference — tabs stripped."""
        mock_batch_infer.return_value = [
            {"entailment": 0.85, "neutral": 0.1, "contradiction": 0.05}
        ]

        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu")
            classifier._label_order = ["contradiction", "entailment", "neutral"]

        dirty_passage = MockPassage(
            text=(
                "Henry was continually on the verge of ruin .\t"
                "Francis I of France\tFrancis I of France\tHoly Roman Emperor Charles V"
            )
        )
        retrievals = [MockRetrievalResult(0.9, 1, dirty_passage)]

        classifier.classify("Henry had financial problems.", retrievals)

        # The passage text must be cleaned in-place before _batch_infer is called
        assert "\t" not in dirty_passage.text
        assert dirty_passage.text == "Henry was continually on the verge of ruin."

    @patch.object(StanceClassifier, "_batch_infer")
    def test_to_dict_is_serialisable(self, mock_batch_infer):
        """StanceResult.to_dict() must return a plain dict with no custom objects."""
        import json

        mock_batch_infer.return_value = [
            {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
        ]

        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu")
            classifier._label_order = ["contradiction", "entailment", "neutral"]

        retrievals = [MockRetrievalResult(0.9, 1, MockPassage("The sky is blue."))]
        result = classifier.classify("The sky is blue.", retrievals)

        d = result.to_dict()
        # Must be JSON-serialisable (no StanceLabel enums, no dataclasses)
        serialised = json.dumps(d)
        assert '"aggregate_label"' in serialised
        assert '"SUPPORTING"' in serialised

    @patch.object(StanceClassifier, "_batch_infer")
    def test_top_supporting_sorted_by_confidence(self, mock_batch_infer):
        """top_supporting must return SUPPORTING passages sorted by confidence desc."""
        mock_batch_infer.return_value = [
            {"entailment": 0.7,  "neutral": 0.2,  "contradiction": 0.1},
            {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
        ]

        with patch("src.claim_processing.stance_classifier.AutoModelForSequenceClassification"), \
             patch("src.claim_processing.stance_classifier.AutoTokenizer"):
            classifier = StanceClassifier(device="cpu")
            classifier._label_order = ["contradiction", "entailment", "neutral"]

        retrievals = [
            MockRetrievalResult(0.8, 1, MockPassage("Lower confidence support.")),
            MockRetrievalResult(0.9, 2, MockPassage("Higher confidence support.")),
        ]
        result = classifier.classify("Claim.", retrievals)

        top = result.top_supporting
        assert len(top) == 2
        assert top[0].confidence > top[1].confidence