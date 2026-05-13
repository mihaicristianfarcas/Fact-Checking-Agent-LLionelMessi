import { motion } from 'framer-motion';
import { SchematicBox, BoxLabel } from './SchematicBox';

const LABELS = [
	{ name: 'SUPPORTS', value: 0.97, color: '#0d9488', active: true },
	{ name: 'REFUTES', value: 0.02, color: '#b91c1c', active: false },
	{ name: 'NEUTRAL', value: 0.01, color: '#64748b', active: false },
];

export function ClaimProcessingDiagram() {
	return (
		<div className="w-full flex flex-col gap-5">
			{/* Step 1: Decompose */}
			<div>
				<div className="flex items-center gap-2 mb-2">
					<span
						className="mono text-[10px] uppercase tracking-[0.18em] px-1.5 py-0.5"
						style={{ background: '#6366f1', color: 'white', fontWeight: 700 }}
					>
						01
					</span>
					<BoxLabel>decompose · compound → atomic</BoxLabel>
				</div>
				<div className="flex items-stretch gap-3">
					<motion.div
						initial={{ opacity: 0, x: -6 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ delay: 0.1 }}
						className="flex-1"
					>
						<SchematicBox variant="solid" className="px-3 py-2 rounded-sm h-full flex flex-col justify-center">
							<BoxLabel>compound</BoxLabel>
							<p className="mono text-[12px] text-[#0b2447] mt-1 leading-snug">
								"Obama, elected in 2008, signed the ACA in 2010."
							</p>
						</SchematicBox>
					</motion.div>

					<div className="flex items-center justify-center w-8">
						<svg width="32" height="12" viewBox="0 0 32 12">
							<motion.line
								x1={0}
								x2={22}
								y1={6}
								y2={6}
								stroke="#0b2447"
								strokeWidth={1.2}
								initial={{ pathLength: 0 }}
								animate={{ pathLength: 1 }}
								transition={{ delay: 0.25, duration: 0.3 }}
							/>
							<polygon points="22,2 30,6 22,10" fill="#0b2447" />
						</svg>
					</div>

					<div className="flex-1 flex flex-col gap-1.5">
						{[
							{ id: 'c1', text: 'Obama was elected in 2008.' },
							{ id: 'c2', text: 'Obama signed the ACA in 2010.' },
						].map((c, i) => (
							<motion.div
								key={c.id}
								initial={{ opacity: 0, y: 4 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ delay: 0.35 + i * 0.08 }}
							>
								<SchematicBox variant="accent" className="px-3 py-1.5 rounded-sm">
									<div className="flex items-center gap-2">
										<span className="mono text-[9px] uppercase tracking-[0.16em] text-[#0369a1] font-bold">
											{c.id}
										</span>
										<p className="mono text-[12px] text-[#0b2447] leading-snug">{c.text}</p>
									</div>
								</SchematicBox>
							</motion.div>
						))}
					</div>

					<div className="w-[24%]">
						<SchematicBox variant="ghost" className="px-2.5 py-2 rounded-sm h-full">
							<BoxLabel>impl</BoxLabel>
							<p className="mono text-[10px] text-[#0b2447] mt-1 leading-snug">
								ollama LLM
								<br />
								<span className="text-[#64748b]">→ rule-based fallback</span>
							</p>
						</SchematicBox>
					</div>
				</div>
			</div>

			{/* Divider */}
			<div className="flex items-center gap-3">
				<div className="flex-1 h-px bg-[#cbd5e1]" />
				<span className="mono text-[9px] uppercase tracking-[0.22em] text-[#64748b]">
					for each (atomic claim, retrieved passage)
				</span>
				<div className="flex-1 h-px bg-[#cbd5e1]" />
			</div>

			{/* Step 2: Stance */}
			<div>
				<div className="flex items-center gap-2 mb-2">
					<span
						className="mono text-[10px] uppercase tracking-[0.18em] px-1.5 py-0.5"
						style={{ background: '#6366f1', color: 'white', fontWeight: 700 }}
					>
						02
					</span>
					<BoxLabel>stance · 3-way NLI per passage</BoxLabel>
				</div>
				<div className="flex items-stretch gap-3">
					{/* Inputs */}
					<div className="flex-1 flex flex-col gap-1.5">
						<motion.div initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.55 }}>
							<SchematicBox variant="solid" className="px-3 py-1.5 rounded-sm">
								<BoxLabel>claim</BoxLabel>
								<p className="mono text-[12px] text-[#0b2447] mt-0.5 leading-snug">
									"Obama was the 44th US President."
								</p>
							</SchematicBox>
						</motion.div>
						<motion.div initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.65 }}>
							<SchematicBox variant="solid" className="px-3 py-1.5 rounded-sm">
								<BoxLabel>passage · Barack_Obama</BoxLabel>
								<p className="mono text-[11px] text-[#0b2447] mt-0.5 leading-snug">
									"Barack Obama served as the 44th president from 2009 to 2017."
								</p>
							</SchematicBox>
						</motion.div>
					</div>

					{/* Model */}
					<div className="flex items-center justify-center w-[16%]">
						<motion.div
							initial={{ opacity: 0, scale: 0.95 }}
							animate={{ opacity: 1, scale: 1 }}
							transition={{ delay: 0.75 }}
						>
							<SchematicBox variant="accent" corners className="px-3 py-2.5 rounded-sm flex flex-col items-center gap-0.5">
								<BoxLabel>model</BoxLabel>
								<span className="mono text-[12px] font-bold text-[#0369a1]">DeBERTa-v3</span>
								<span className="mono text-[9px] uppercase tracking-[0.12em] text-[#64748b]">3-way NLI</span>
							</SchematicBox>
						</motion.div>
					</div>

					{/* Outputs */}
					<div className="flex-1 flex flex-col gap-1 justify-center">
						<BoxLabel>softmax</BoxLabel>
						{LABELS.map((l, i) => (
							<motion.div
								key={l.name}
								initial={{ opacity: 0, x: 6 }}
								animate={{ opacity: 1, x: 0 }}
								transition={{ delay: 0.85 + i * 0.08 }}
								className="flex items-center gap-2"
							>
								<span
									className="mono text-[10px] uppercase tracking-[0.14em] w-16"
									style={{ color: l.color, fontWeight: l.active ? 800 : 500, opacity: l.active ? 1 : 0.55 }}
								>
									{l.name}
								</span>
								<div className="flex-1 h-2.5 bg-white hairline-soft relative overflow-hidden">
									<motion.div
										initial={{ width: 0 }}
										animate={{ width: `${l.value * 100}%` }}
										transition={{ delay: 0.95 + i * 0.08, duration: 0.5, ease: 'easeOut' }}
										className="h-full"
										style={{ background: l.color, opacity: l.active ? 1 : 0.4 }}
									/>
								</div>
								<span
									className="mono text-[11px] tabular-nums w-9 text-right"
									style={{ color: l.color, fontWeight: l.active ? 800 : 500 }}
								>
									{l.value.toFixed(2)}
								</span>
							</motion.div>
						))}
					</div>
				</div>
			</div>

			{/* Footer caveat */}
			<motion.p
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 1.2 }}
				className="mono text-[10px] text-[#64748b] uppercase tracking-[0.16em] text-center"
			>
				every classification carries the passage id · citations cannot escape the retrieved set
			</motion.p>
		</div>
	);
}
