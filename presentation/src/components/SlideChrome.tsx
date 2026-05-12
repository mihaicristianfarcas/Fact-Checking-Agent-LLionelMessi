import type { ReactNode } from 'react';

type SectionId = 'INTRO' | 'A' | 'B' | 'C' | 'D' | 'DEMO' | 'END';

const SECTION_LABEL: Record<SectionId, string> = {
	INTRO: '§ 0 — OVERVIEW',
	A: '§ A — DATA & INGESTION',
	B: '§ B — CLAIM PROCESSING',
	C: '§ C — MODEL TRAINING',
	D: '§ D — SCORING · SYNTHESIS · EVAL',
	DEMO: '§ DEMO',
	END: '§ END',
};

export function SectionLabel({ section }: { section: SectionId }) {
	return (
		<span
			className="mono text-[11px] tracking-[0.18em] uppercase text-[#0b2447]/70"
			style={{ fontWeight: 600 }}
		>
			{SECTION_LABEL[section]}
		</span>
	);
}

export function FigureLabel({ id, title }: { id: string; title: string }) {
	return (
		<span
			className="mono text-[11px] tracking-[0.18em] uppercase text-[#0b2447]/60"
			style={{ fontStyle: 'italic', fontWeight: 500 }}
		>
			FIG. {id} · {title}
		</span>
	);
}

export function SlideHeader({
	section,
	figureId,
	figureTitle,
	heading,
	kicker,
}: {
	section: SectionId;
	figureId: string;
	figureTitle: string;
	heading: string;
	kicker?: string;
}) {
	return (
		<div className="shrink-0">
			<div className="flex items-center justify-between mb-6">
				<SectionLabel section={section} />
				<FigureLabel id={figureId} title={figureTitle} />
			</div>
			<div className="border-t border-[#0b2447] mb-5" />
			{kicker && (
				<p className="mono text-xs uppercase tracking-[0.22em] text-[#0ea5e9] font-semibold mb-3">
					{kicker}
				</p>
			)}
			<h2 className="text-[44px] leading-[1.05] font-bold text-[#0b2447] tracking-tight">
				{heading}
			</h2>
		</div>
	);
}

export function SlideShell({ children }: { children: ReactNode }) {
	return <div className="w-full h-full flex flex-col">{children}</div>;
}
