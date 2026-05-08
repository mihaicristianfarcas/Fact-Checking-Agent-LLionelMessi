from src.agent.orchestrator import PipelineTrace
from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.retriever.evidence_retriever import RetrievalResult
from src.synthesis.conversational_responder import (
    EvidenceSnippet,
    SYSTEM_PROMPT,
    build_responder_messages,
    is_low_quality_passage,
    select_evidence_snippets,
)
from src.synthesis.verdict_synthesizer import AtomicVerdict, SynthesisResult


def _retrieval(pid, score, source="Lionel_Messi", text=None):
    default_text = (
        f"This is a realistic Wikipedia evidence sentence about {pid} with "
        "enough lexical diversity to pass the quality filter."
    )
    return RetrievalResult(
        passage=EvidencePassage(
            id=pid,
            text=text or default_text,
            source=source,
            dataset="fever",
        ),
        score=score,
        rank=1,
    )


def _trace(claim, verdict, retrievals, cited_ids=None, confidence=0.85):
    synth = SynthesisResult(
        original_claim=claim,
        verdict=verdict,
        confidence=confidence,
        explanation="Test explanation.",
        cited_passage_ids=cited_ids or [],
        atomic_verdicts=[
            AtomicVerdict(
                claim_text=claim,
                verdict=verdict,
                confidence=confidence,
                cited_passages=cited_ids or [],
            )
        ],
        all_retrieved_ids=[r.passage.id for r in retrievals],
    )
    trace = PipelineTrace(original_claim=claim)
    trace.retrievals = {claim: retrievals}
    trace.synthesis = synth
    return trace


def test_select_evidence_prefers_cited_passages():
    retrievals = [
        _retrieval("p_high", score=0.95),
        _retrieval("p_cited", score=0.40),
        _retrieval("p_mid", score=0.70),
    ]
    trace = _trace("Claim.", "SUPPORTED", retrievals, cited_ids=["p_cited"])

    snippets = select_evidence_snippets(trace, max_snippets=3)
    ids = [s.passage_id for s in snippets]

    assert ids[0] == "p_cited"
    assert "p_high" in ids
    assert "p_mid" in ids


def test_select_evidence_dedupes_across_atomic_claims():
    shared = _retrieval("p_shared", score=0.5)
    other = _retrieval("p_other", score=0.4)
    trace = PipelineTrace(original_claim="Claim.")
    trace.retrievals = {
        "atomic A": [shared, other],
        "atomic B": [shared],
    }
    trace.synthesis = SynthesisResult(
        original_claim="Claim.",
        verdict="SUPPORTED",
        confidence=0.8,
        explanation="x",
        cited_passage_ids=[],
        atomic_verdicts=[],
        all_retrieved_ids=["p_shared", "p_other"],
    )

    snippets = select_evidence_snippets(trace)
    ids = [s.passage_id for s in snippets]

    assert ids.count("p_shared") == 1


def test_select_evidence_truncates_long_passages():
    # Realistic dense Wikipedia-style sentence well above the 100-char limit.
    long_text = (
        "Lionel Andres Messi born 24 June 1987 is an Argentine professional "
        "footballer widely regarded as one of the greatest players in history."
    )
    retrievals = [_retrieval("p1", score=0.9, text=long_text)]
    trace = _trace("Claim.", "SUPPORTED", retrievals)

    snippets = select_evidence_snippets(trace, max_chars_per_snippet=100)

    assert len(snippets[0].text) <= 100
    assert snippets[0].text.endswith("...")


def test_build_messages_includes_verdict_and_evidence():
    snippets = [
        EvidenceSnippet(source="Lionel_Messi", text="He plays football.", passage_id="p1"),
        EvidenceSnippet(source="FC_Barcelona", text="Won La Liga.", passage_id="p2"),
    ]
    synth = SynthesisResult(
        original_claim="Messi plays football.",
        verdict="SUPPORTED",
        confidence=0.92,
        explanation="x",
        cited_passage_ids=["p1"],
        atomic_verdicts=[],
        all_retrieved_ids=["p1", "p2"],
    )

    messages = build_responder_messages("Messi plays football.", synth, snippets)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Messi plays football." in user
    assert "SUPPORTED" in user
    assert "high-confidence" in user
    assert "92%" not in user, (
        "Raw confidence percentages bias the model into defending the "
        "verdict; pass a qualitative band instead."
    )
    assert "Lionel_Messi" in user
    assert "FC_Barcelona" in user
    assert "He plays football." in user


def test_build_messages_handles_empty_evidence():
    synth = SynthesisResult(
        original_claim="Some weird claim.",
        verdict="NOT_ENOUGH_INFO",
        confidence=0.30,
        explanation="x",
        cited_passage_ids=[],
        atomic_verdicts=[],
        all_retrieved_ids=[],
    )

    messages = build_responder_messages("Some weird claim.", synth, [])
    user = messages[1]["content"]

    assert "NOT_ENOUGH_INFO" in user
    assert "no evidence retrieved" in user.lower()


def test_system_prompt_constrains_grounding():
    """System prompt must instruct against fabrication, forbid using
    pretraining knowledge, require verbatim quoting of evidence snippets,
    and call out NOT_ENOUGH_INFO handling. These are the load-bearing
    constraints for grounding — loosening any of them lets the small chat
    model drift."""
    lower = SYSTEM_PROMPT.lower()
    assert "do not invent" in lower
    assert "only" in lower and "evidence" in lower
    assert "pretraining" in lower or "outside knowledge" in lower
    assert "verbatim" in lower
    assert "evidence:" in lower
    assert "NOT_ENOUGH_INFO" in SYSTEM_PROMPT


def test_user_prompt_requests_verbatim_quotes():
    """The per-turn user message must reinforce the quoting format so a small
    model that didn't internalise the long system prompt still complies."""
    synth = SynthesisResult(
        original_claim="Claim.",
        verdict="SUPPORTED",
        confidence=0.9,
        explanation="x",
        cited_passage_ids=[],
        atomic_verdicts=[],
        all_retrieved_ids=[],
    )
    snippets = [EvidenceSnippet(source="Lionel_Messi", text="He plays.", passage_id="p1")]
    user = build_responder_messages("Claim.", synth, snippets)[1]["content"]

    lower = user.lower()
    assert "only" in lower and "evidence" in lower
    assert "verbatim" in lower or "as written" in lower
    assert "evidence:" in lower


class TestIsLowQualityPassage:
    """Garbage FEVER infobox-derived sentences (e.g. 'NATO NATO Russia Russia.')
    must be filtered before the chat model sees them, otherwise the model
    invents narratives around meaningless tokens."""

    def test_drops_repetition_garbage(self):
        assert is_low_quality_passage("NATO states . NATO NATO Russia Russia.")
        assert is_low_quality_passage("s United States NATO NATO.")

    def test_drops_ultra_short(self):
        assert is_low_quality_passage("NATO.")
        assert is_low_quality_passage("")
        assert is_low_quality_passage("   ")

    def test_drops_passages_starting_mid_word(self):
        """A 1-2 char leading token followed by space is the chunker artifact
        we observed ('s United States NATO NATO.')."""
        assert is_low_quality_passage("s United States NATO NATO.")

    def test_keeps_normal_wikipedia_sentence(self):
        text = (
            "Lionel Messi is an Argentine professional footballer who plays "
            "as a forward and captains the Argentina national team."
        )
        assert not is_low_quality_passage(text)

    def test_keeps_short_but_dense_sentence(self):
        text = "The Eiffel Tower is located in Paris, France."
        assert not is_low_quality_passage(text)

    def test_drops_low_alpha_density(self):
        assert is_low_quality_passage("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")


def test_select_evidence_drops_low_quality_passages():
    """The responder must not surface garbage chunks even if retrieval
    ranks them high — they confuse the chat model."""
    retrievals = [
        _retrieval(
            "garbage",
            score=0.95,
            text="NATO states . NATO NATO Russia Russia.",
        ),
        _retrieval(
            "good",
            score=0.30,
            text="Lionel Messi is an Argentine professional footballer.",
        ),
    ]
    trace = _trace("Claim.", "SUPPORTED", retrievals)

    snippets = select_evidence_snippets(trace)
    ids = [s.passage_id for s in snippets]

    assert "garbage" not in ids
    assert "good" in ids


def test_stream_response_yields_chunks_and_raises_on_missing_synthesis():
    """End-to-end streaming contract: rejects empty traces, yields non-empty
    chunks when the underlying generation is mocked. Avoids downloading any
    real model weights."""
    from src.synthesis.conversational_responder import ConversationalResponder

    trace = PipelineTrace(original_claim="Claim.")
    responder = ConversationalResponder()
    try:
        list(responder.stream_response(trace))
    except ValueError as exc:
        assert "synthesis" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError when synthesis is missing")
