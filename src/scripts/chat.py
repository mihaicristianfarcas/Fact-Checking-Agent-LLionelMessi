#!/usr/bin/env python3
"""Interactive fact-checking chat.

Streams a conversational reply for each claim the user types. The structured
verification (decompose -> retrieve -> stance -> synthesize) is unchanged;
the trained DeBERTa FEVER verifier replaces the synthesizer's verdict, and
a small local chat model narrates the result.

Usage:
    python -m src.scripts.chat
    python -m src.scripts.chat --top-k 8
    python -m src.scripts.chat --no-verifier      # disable trained verifier
    python -m src.scripts.chat --no-trace         # hide verdict line
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from src.agent.orchestrator import FactCheckAgent, PipelineTrace
from src.synthesis.conversational_responder import (
    DEFAULT_MODEL_ID,
    ConversationalResponder,
)

console = Console()

DEFAULT_VERIFIER_MODEL = "vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned"

VERDICT_STYLE = {
    "SUPPORTED": "bold green",
    "REFUTED": "bold red",
    "NOT_ENOUGH_INFO": "bold yellow",
}


def _print_verdict_line(synthesis) -> None:
    color = VERDICT_STYLE.get(synthesis.verdict, "white")
    console.print(
        f"[dim]Pipeline:[/dim] [{color}]{synthesis.verdict}[/{color}] "
        f"[dim](confidence {synthesis.confidence:.0%})[/dim]"
    )


def _apply_trained_verifier(trace: PipelineTrace, verifier) -> PipelineTrace:
    """Override trace.synthesis with the trained DeBERTa verifier's verdict.

    Mirrors the integration in src/scripts/evaluate_pipeline.py: flatten
    retrievals -> predict -> override the synthesis result. Calibration
    thresholds and the baseline-refute fallback are intentionally left at
    their defaults; surface-level threshold tuning is queued for the CUDA
    re-run, and the chat REPL should reflect the verifier's raw behaviour.
    """
    from src.claim_processing.verdict_verifier import (
        flatten_trace_retrievals,
        override_synthesis_with_prediction,
    )

    if trace.synthesis is None:
        return trace

    retrievals = flatten_trace_retrievals(trace)
    if not retrievals:
        return trace

    prediction = verifier.predict(trace.original_claim, retrievals)
    trace.synthesis = override_synthesis_with_prediction(
        trace.synthesis, prediction, retrievals
    )
    trace.steps_executed.append("trained_verifier")
    return trace


def chat_loop(
    agent: FactCheckAgent,
    responder: ConversationalResponder,
    *,
    verifier=None,
    show_trace: bool = True,
) -> None:
    console.print(
        Panel(
            "[bold]Fact-Checking Chat[/bold]\n"
            "Type a claim and I'll verify it, then explain the result.\n"
            "[dim]Commands: /quit, /exit, Ctrl-D[/dim]",
            border_style="bright_blue",
        )
    )

    while True:
        try:
            claim = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not claim:
            continue
        if claim.lower() in {"/quit", "/exit", ":q"}:
            console.print("[dim]bye[/dim]")
            return

        try:
            trace = agent.check_with_trace(claim)
            if verifier is not None:
                trace = _apply_trained_verifier(trace, verifier)
        except Exception as exc:
            console.print(f"[red]pipeline error:[/red] {exc}")
            continue

        if show_trace:
            _print_verdict_line(trace.synthesis)

        console.print("[bold magenta]agent ›[/bold magenta] ", end="")
        try:
            for chunk in responder.stream_response(trace):
                sys.stdout.write(chunk)
                sys.stdout.flush()
        except Exception as exc:
            console.print(f"\n[red]responder error:[/red] {exc}")
            continue
        sys.stdout.write("\n\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive fact-checking chat.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Passages to retrieve per atomic claim.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive retrieval (extra passages on low scores).",
    )
    parser.add_argument(
        "--responder-model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace ID of the local chat model used for narration.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=280,
        help="Maximum tokens in each streamed reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, deterministic, recommended "
        "for grounding). Higher values increase fluency at the cost of "
        "fabrication risk.",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=DEFAULT_VERIFIER_MODEL,
        help="HuggingFace ID or local path of the trained DeBERTa FEVER "
        "verifier. The chat REPL uses it to override the synthesizer's "
        "verdict (matching the eval pipeline).",
    )
    parser.add_argument(
        "--verifier-device",
        type=str,
        default=None,
        help="Optional verifier device override: cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        help="Skip loading the trained verifier and use the synthesizer's "
        "raw verdict (for ablation only).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Hide the verdict/confidence line above each reply.",
    )
    args = parser.parse_args()

    console.print("[dim]Loading verification pipeline...[/dim]")
    agent = FactCheckAgent(top_k=args.top_k, adaptive=args.adaptive)

    verifier = None
    if not args.no_verifier:
        from src.claim_processing.verdict_verifier import FeverVerdictVerifier

        console.print(
            f"[dim]Loading trained DeBERTa verifier ({args.verifier_model})...[/dim]"
        )
        verifier = FeverVerdictVerifier(
            args.verifier_model,
            device=args.verifier_device,
        )

    console.print(
        f"[dim]Loading responder model ({args.responder_model})...[/dim]"
    )
    responder = ConversationalResponder(
        model_id=args.responder_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    chat_loop(
        agent, responder, verifier=verifier, show_trace=not args.no_trace
    )


if __name__ == "__main__":
    main()
