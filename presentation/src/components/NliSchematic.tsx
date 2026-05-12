import { motion } from 'framer-motion';
import { SchematicBox, BoxLabel } from './SchematicBox';

const LABELS = [
	{ name: 'SUPPORTS', value: 0.97, color: '#0d9488', active: true },
	{ name: 'REFUTES', value: 0.02, color: '#b91c1c', active: false },
	{ name: 'NEUTRAL', value: 0.01, color: '#64748b', active: false },
];

export function NliSchematic() {
	return (
		<div className="flex items-stretch gap-5 w-full">
			{/* Inputs (claim + passage) */}
			<div className="flex flex-col gap-3 w-[40%]">
				<motion.div initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
					<SchematicBox variant="solid" className="px-4 py-3 rounded-sm">
						<BoxLabel>input · claim</BoxLabel>
						<p className="mono text-[13px] text-[#0b2447] mt-1 leading-snug">
							"Obama was the 44th US President."
						</p>
					</SchematicBox>
				</motion.div>
				<motion.div initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
					<SchematicBox variant="solid" className="px-4 py-3 rounded-sm">
						<BoxLabel>input · passage</BoxLabel>
						<p className="mono text-[12px] text-[#0b2447] mt-1 leading-snug">
							"Barack Obama served as the 44th president from 2009 to 2017."
						</p>
						<p className="mono text-[10px] text-[#64748b] mt-1.5 uppercase tracking-[0.12em]">
							source: <span style={{ color: '#0b2447', fontWeight: 600 }}>Barack_Obama</span>
						</p>
					</SchematicBox>
				</motion.div>
			</div>

			{/* Arrow + model */}
			<div className="flex flex-col items-center justify-center w-[18%]">
				<svg width="40" height="12" viewBox="0 0 40 12">
					<motion.line
						x1={0}
						x2={28}
						y1={6}
						y2={6}
						stroke="#0b2447"
						strokeWidth={1.2}
						initial={{ pathLength: 0 }}
						animate={{ pathLength: 1 }}
						transition={{ delay: 0.3, duration: 0.3 }}
					/>
					<polygon points="28,2 36,6 28,10" fill="#0b2447" />
				</svg>
				<motion.div
					initial={{ opacity: 0, scale: 0.95 }}
					animate={{ opacity: 1, scale: 1 }}
					transition={{ delay: 0.4 }}
					className="my-2"
				>
					<SchematicBox variant="accent" corners className="px-4 py-4 rounded-sm flex flex-col items-center gap-1.5">
						<BoxLabel>model</BoxLabel>
						<span className="mono text-base font-bold text-[#0369a1]">DeBERTa-v3</span>
						<span className="mono text-[10px] uppercase tracking-[0.14em] text-[#64748b]">3-way NLI</span>
					</SchematicBox>
				</motion.div>
				<svg width="40" height="12" viewBox="0 0 40 12">
					<motion.line
						x1={0}
						x2={28}
						y1={6}
						y2={6}
						stroke="#0b2447"
						strokeWidth={1.2}
						initial={{ pathLength: 0 }}
						animate={{ pathLength: 1 }}
						transition={{ delay: 0.55, duration: 0.3 }}
					/>
					<polygon points="28,2 36,6 28,10" fill="#0b2447" />
				</svg>
			</div>

			{/* Outputs */}
			<div className="flex flex-col gap-2 w-[42%] justify-center">
				<BoxLabel>output · softmax</BoxLabel>
				{LABELS.map((l, i) => (
					<motion.div
						key={l.name}
						initial={{ opacity: 0, x: 8 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ delay: 0.65 + i * 0.1 }}
						className="flex items-center gap-3"
					>
						<span
							className="mono text-[11px] uppercase tracking-[0.16em] w-24"
							style={{ color: l.color, fontWeight: l.active ? 800 : 500, opacity: l.active ? 1 : 0.55 }}
						>
							{l.name}
						</span>
						<div className="flex-1 h-3 bg-white hairline-soft relative overflow-hidden">
							<motion.div
								initial={{ width: 0 }}
								animate={{ width: `${l.value * 100}%` }}
								transition={{ delay: 0.8 + i * 0.1, duration: 0.5, ease: 'easeOut' }}
								className="h-full"
								style={{ background: l.color, opacity: l.active ? 1 : 0.4 }}
							/>
						</div>
						<span
							className="mono text-[12px] tabular-nums w-12 text-right"
							style={{ color: l.color, fontWeight: l.active ? 800 : 500 }}
						>
							{l.value.toFixed(2)}
						</span>
					</motion.div>
				))}
			</div>
		</div>
	);
}
