#!/usr/bin/env python3
"""Interactive fact-checking demo.

One command to run the full pipeline on a claim and print the result.

Usage:
    # Check a single claim
    python -m src.scripts.run_demo --claim "The Earth is flat."

    # Run the built-in demo set (supported, refuted, abstention)
    python -m src.scripts.run_demo --demo

    # Save the full trace as JSON
    python -m src.scripts.run_demo --claim "Some claim" --trace-out trace.json

    # Adjust retrieval depth
    python -m src.scripts.run_demo --claim "Some claim" --top-k 10
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.agent.orchestrator import FactCheckAgent

console = Console()

DEMO_CLAIMS = [
    {
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "expected": "SUPPORTED",
        "note": "FEVER gold: SUPPORTED - evidence exists in Wikipedia.",
    },
    {
        "claim": "The Eiffel Tower is located in Berlin.",
        "expected": "REFUTED",
        "note": "The Eiffel Tower is in Paris - evidence should contradict.",
    },
    {
        "claim": "A secret underwater civilization was discovered near Antarctica in 2024.",
        "expected": "NOT_ENOUGH_INFO",
        "note": "Fabricated claim - corpus should have no relevant evidence.",
    },
]


def print_result(result, claim_text: str, note: str = "") -> None:
    """Pretty-print a SynthesisResult using rich."""
    verdict_colors = {
        "SUPPORTED": "bold green",
        "REFUTED": "bold red",
        "NOT_ENOUGH_INFO": "bold yellow",
    }
    color = verdict_colors.get(result.verdict, "white")

    header = Text()
    header.append("Verdict: ", style="bold")
    header.append(result.verdict, style=color)
    header.append(f"  (confidence: {result.confidence:.0%})")

    console.print()
    console.print(Panel(
        f"[bold]Claim:[/bold] {claim_text}",
        title="Fact-Check",
        border_style="blue",
    ))
    console.print(header)
    console.print()
    console.print(f"[bold]Explanation:[/bold] {escape(result.explanation)}")

    if result.cited_passage_ids:
        console.print(f"\n[bold]Citations:[/bold] {', '.join(result.cited_passage_ids)}")

    if result.atomic_verdicts and len(result.atomic_verdicts) > 1:
        table = Table(title="Atomic Claims Breakdown", show_lines=True)
        table.add_column("Sub-claim", style="cyan", max_width=60)
        table.add_column("Verdict", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Citations", max_width=30)

        for av in result.atomic_verdicts:
            v_style = verdict_colors.get(av.verdict, "white")
            table.add_row(
                av.claim_text,
                Text(av.verdict, style=v_style),
                f"{av.confidence:.0%}",
                ", ".join(av.cited_passages) if av.cited_passages else "-",
            )
        console.print(table)

    if result.hallucinated_citations:
        console.print(
            f"\n[bold red]WARNING:[/bold red] Hallucinated citations detected: "
            f"{result.hallucinated_citations}"
        )

    if note:
        console.print(f"\n[dim]{note}[/dim]")

    console.print()


def main():
    parser = argparse.ArgumentParser(description="Fact-Checking Agent Demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--claim",
        type=str,
        help="A claim to fact-check.",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in demo set (3 curated claims).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Passages to retrieve per atomic claim.",
    )
    parser.add_argument(
        "--trace-out",
        type=str,
        default=None,
        help="Path to save the full execution trace as JSON.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive retrieval.",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold]Automated Fact-Checking Agent[/bold]\n"
            "Decompose -> Retrieve -> Stance -> Score -> Synthesize",
            title="LLionelMessi",
            border_style="bright_blue",
        )
    )
    console.print("[dim]Loading models (first run may take a minute)...[/dim]")

    agent = FactCheckAgent(top_k=args.top_k, adaptive=args.adaptive)

    if args.demo:
        console.print(
            f"\n[bold]Running {len(DEMO_CLAIMS)} demo claims...[/bold]\n"
        )
        traces = []
        for demo in DEMO_CLAIMS:
            trace = agent.check_with_trace(demo["claim"])
            print_result(
                trace.synthesis,
                demo["claim"],
                note=f"Expected: {demo['expected']} - {demo['note']}",
            )
            traces.append(trace)

        if args.trace_out:
            out = Path(args.trace_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    [t.to_dict() for t in traces],
                    indent=2,
                    default=str,
                )
            )
            console.print(f"[dim]Traces saved to {out}[/dim]")
    else:
        trace = agent.check_with_trace(args.claim)
        print_result(trace.synthesis, args.claim)

        if args.trace_out:
            out = Path(args.trace_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(trace.to_dict(), indent=2, default=str)
            )
            console.print(f"[dim]Trace saved to {out}[/dim]")

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
