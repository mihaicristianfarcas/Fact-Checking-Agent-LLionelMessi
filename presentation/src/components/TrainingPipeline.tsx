import { motion } from 'framer-motion';
import { SchematicBox, BoxLabel } from './SchematicBox';
import { Database, Cpu, Scale, Package } from 'lucide-react';

const STAGES = [
	{ id: '01', label: 'data', title: '126K triples', sub: 'claim · evidence · verdict', icon: Database },
	{ id: '02', label: 'sft', title: 'TinyLlama-1.1B + LoRA', sub: 'teach the verdict format', icon: Cpu },
	{ id: '03', label: 'pairs', title: 'preference pairs', sub: 'well-cited > unsupported', icon: Scale },
	{ id: '04', label: 'dpo', title: 'DPO', sub: 'conservative verdicts', icon: Cpu },
	{ id: '05', label: 'output', title: 'LoRA adapter', sub: 'andreiungureanu/…', icon: Package },
];

export function TrainingPipeline() {
	return (
		<div className="flex items-stretch justify-between gap-2 w-full">
			{STAGES.map((s, i) => {
				const Icon = s.icon;
				return (
					<div key={s.id} className="flex items-stretch gap-2 flex-1">
						<motion.div
							initial={{ opacity: 0, y: 8 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ delay: 0.1 + i * 0.1 }}
							className="flex-1"
						>
							<SchematicBox
								variant={i === STAGES.length - 1 ? 'accent' : 'solid'}
								className="px-3 py-3 rounded-sm flex flex-col gap-1.5 h-full"
							>
								<div className="flex items-center justify-between">
									<BoxLabel>{s.id}</BoxLabel>
									<Icon size={14} className="text-[#0b2447]/60" strokeWidth={1.5} />
								</div>
								<span
									className="mono text-[9px] uppercase tracking-[0.16em] text-[#0ea5e9] font-bold"
								>
									{s.label}
								</span>
								<span className="text-[13px] font-bold text-[#0b2447] leading-tight">{s.title}</span>
								<span className="mono text-[10px] text-[#64748b] leading-tight">{s.sub}</span>
							</SchematicBox>
						</motion.div>
						{i < STAGES.length - 1 && (
							<div className="flex items-center w-4">
								<svg width="16" height="10" viewBox="0 0 16 10">
									<motion.line
										x1={0}
										x2={10}
										y1={5}
										y2={5}
										stroke="#0b2447"
										strokeWidth={1.2}
										initial={{ pathLength: 0 }}
										animate={{ pathLength: 1 }}
										transition={{ delay: 0.4 + i * 0.1, duration: 0.25 }}
									/>
									<polygon points="10,1 14,5 10,9" fill="#0b2447" />
								</svg>
							</div>
						)}
					</div>
				);
			})}
		</div>
	);
}
