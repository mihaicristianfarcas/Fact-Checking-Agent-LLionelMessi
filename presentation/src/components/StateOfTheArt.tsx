import { motion } from 'framer-motion';
import { Check, X, Minus } from 'lucide-react';
import { SchematicBox, BoxLabel } from './SchematicBox';

type Cell = { kind: 'yes' | 'no' | 'partial'; label?: string };

type Row = {
	system: string;
	sub: string;
	approach: string;
	citations: Cell;
	calibration: Cell;
	ours?: boolean;
};

const ROWS: Row[] = [
	{
		system: 'FEVER baselines',
		sub: 'DrQA · ESIM',
		approach: 'retrieve + classify',
		citations: { kind: 'partial', label: 'retrieval ids' },
		calibration: { kind: 'no' },
	},
	{
		system: 'LLM-only',
		sub: 'GPT / Llama prompted',
		approach: 'generative verdict',
		citations: { kind: 'no', label: 'hallucinated' },
		calibration: { kind: 'no' },
	},
	{
		system: 'LLM + retrieval',
		sub: 'Factool · Loki · ReAct',
		approach: 'tool-use, free-form cite',
		citations: { kind: 'partial', label: 'best-effort' },
		calibration: { kind: 'no' },
	},
	{
		system: 'Ours',
		sub: 'tool-augmented + trained head',
		approach: 'orchestrator + DeBERTa verifier',
		citations: { kind: 'yes', label: 'restricted by construction' },
		calibration: { kind: 'yes', label: 'temperature-scaled' },
		ours: true,
	},
];

function CellGlyph({ cell }: { cell: Cell }) {
	const color = cell.kind === 'yes' ? '#0d9488' : cell.kind === 'no' ? '#b91c1c' : '#d97706';
	const Icon = cell.kind === 'yes' ? Check : cell.kind === 'no' ? X : Minus;
	return (
		<div className="flex items-center gap-2">
			<span
				className="inline-flex items-center justify-center w-4 h-4 shrink-0"
				style={{ border: `1px solid ${color}`, color }}
			>
				<Icon size={11} strokeWidth={2.5} />
			</span>
			{cell.label && (
				<span
					className="mono text-[11px] leading-tight"
					style={{ color: cell.kind === 'yes' ? '#0b2447' : '#64748b', fontWeight: cell.kind === 'yes' ? 600 : 500 }}
				>
					{cell.label}
				</span>
			)}
		</div>
	);
}

export function StateOfTheArt() {
	return (
		<div className="w-full">
			<SchematicBox variant="solid" className="rounded-sm overflow-hidden">
				{/* Column headers */}
				<div
					className="grid mono text-[10px] uppercase tracking-[0.18em] text-[#64748b] px-4 py-2.5 border-b border-[#0b2447]"
					style={{ gridTemplateColumns: '26% 26% 24% 24%', fontWeight: 700 }}
				>
					<span>system</span>
					<span>approach</span>
					<span>citations</span>
					<span>calibration</span>
				</div>

				{/* Rows */}
				{ROWS.map((r, i) => (
					<motion.div
						key={r.system}
						initial={{ opacity: 0, y: 6 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ delay: 0.1 + i * 0.1 }}
						className="grid items-center px-4 py-3"
						style={{
							gridTemplateColumns: '26% 26% 24% 24%',
							background: r.ours ? '#f0f9ff' : 'transparent',
							borderBottom: i < ROWS.length - 1 ? '1px solid #e2e8f0' : 'none',
							borderLeft: r.ours ? '3px solid #0ea5e9' : '3px solid transparent',
						}}
					>
						<div>
							<p
								className="text-[14px] leading-tight"
								style={{ color: r.ours ? '#0369a1' : '#0b2447', fontWeight: r.ours ? 800 : 700 }}
							>
								{r.system}
							</p>
							<p className="mono text-[10px] uppercase tracking-[0.12em] text-[#64748b] mt-0.5">
								{r.sub}
							</p>
						</div>
						<p
							className="mono text-[12px] leading-snug"
							style={{ color: r.ours ? '#0b2447' : '#0b2447', fontWeight: r.ours ? 600 : 500 }}
						>
							{r.approach}
						</p>
						<CellGlyph cell={r.citations} />
						<CellGlyph cell={r.calibration} />
					</motion.div>
				))}
			</SchematicBox>

			{/* Legend / takeaway */}
			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 0.6 }}
				className="mt-4 flex items-center justify-between gap-6"
			>
				<div className="flex items-center gap-5 mono text-[10px] uppercase tracking-[0.14em] text-[#64748b]">
					<div className="flex items-center gap-1.5">
						<span className="inline-flex items-center justify-center w-3 h-3" style={{ border: '1px solid #0d9488', color: '#0d9488' }}>
							<Check size={9} strokeWidth={2.5} />
						</span>
						<span>guaranteed</span>
					</div>
					<div className="flex items-center gap-1.5">
						<span className="inline-flex items-center justify-center w-3 h-3" style={{ border: '1px solid #d97706', color: '#d97706' }}>
							<Minus size={9} strokeWidth={2.5} />
						</span>
						<span>partial</span>
					</div>
					<div className="flex items-center gap-1.5">
						<span className="inline-flex items-center justify-center w-3 h-3" style={{ border: '1px solid #b91c1c', color: '#b91c1c' }}>
							<X size={9} strokeWidth={2.5} />
						</span>
						<span>absent</span>
					</div>
				</div>
				<BoxLabel>where we differ — verifier head + citation tracing</BoxLabel>
			</motion.div>
		</div>
	);
}
