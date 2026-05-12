import { motion } from 'framer-motion';
import { SchematicBox, BoxLabel } from './SchematicBox';

const STEPS = [
	{ n: '01', name: 'decompose', sub: 'compound → atomic' },
	{ n: '02', name: 'retrieve', sub: 'top-k passages / atomic' },
	{ n: '03', name: 'score_credibility', sub: 'source → trust' },
	{ n: '04', name: 'classify_stance', sub: '(claim, p) → {S/R/N}' },
	{ n: '05', name: 'synthesize', sub: 'aggregate signals' },
	{ n: '06', name: 'verify', sub: 'DeBERTa verifier' },
	{ n: '07', name: 'calibrate', sub: 'temperature scaling' },
];

export function OrchestratorDiagram() {
	return (
		<div className="flex flex-col gap-1.5 w-full">
			{STEPS.map((s, i) => (
				<motion.div
					key={s.n}
					initial={{ opacity: 0, x: -6 }}
					animate={{ opacity: 1, x: 0 }}
					transition={{ delay: 0.08 + i * 0.06 }}
				>
					<SchematicBox
						variant={i === STEPS.length - 1 ? 'accent' : 'solid'}
						className="px-4 py-2 rounded-sm flex items-center gap-4"
					>
						<BoxLabel>{s.n}</BoxLabel>
						<span className="mono text-[13px] font-bold text-[#0b2447] tracking-tight w-44">{s.name}</span>
						<span className="text-[12px] text-[#64748b]">{s.sub}</span>
					</SchematicBox>
				</motion.div>
			))}
		</div>
	);
}
