import { motion } from 'framer-motion';
import { SchematicBox, BoxLabel } from './SchematicBox';

export function DecomposerTree() {
	return (
		<div className="flex flex-col items-center gap-3 w-full">
			{/* Root */}
			<motion.div
				initial={{ opacity: 0, y: -8 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ delay: 0.1 }}
				className="w-full max-w-2xl"
			>
				<SchematicBox variant="solid" className="px-5 py-3 rounded-sm">
					<BoxLabel>compound claim</BoxLabel>
					<p className="mono text-[15px] text-[#0b2447] mt-1 leading-snug">
						"Obama, elected in 2008, signed the ACA in 2010."
					</p>
				</SchematicBox>
			</motion.div>

			{/* Branch lines */}
			<svg width="600" height="40" viewBox="0 0 600 40">
				<motion.path
					d="M 300 0 L 300 16 L 140 16 L 140 38"
					stroke="#0b2447"
					strokeWidth={1.2}
					fill="none"
					initial={{ pathLength: 0 }}
					animate={{ pathLength: 1 }}
					transition={{ delay: 0.3, duration: 0.4 }}
				/>
				<motion.path
					d="M 300 0 L 300 16 L 460 16 L 460 38"
					stroke="#0b2447"
					strokeWidth={1.2}
					fill="none"
					initial={{ pathLength: 0 }}
					animate={{ pathLength: 1 }}
					transition={{ delay: 0.3, duration: 0.4 }}
				/>
			</svg>

			{/* Atomic claims */}
			<div className="flex gap-8 w-full max-w-2xl -mt-1">
				{[
					{ id: 'c1', text: 'Obama was elected in 2008.' },
					{ id: 'c2', text: 'Obama signed the ACA in 2010.' },
				].map((c, i) => (
					<motion.div
						key={c.id}
						initial={{ opacity: 0, y: 8 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ delay: 0.55 + i * 0.1 }}
						className="flex-1"
					>
						<SchematicBox variant="accent" className="px-4 py-3 rounded-sm">
							<div className="flex items-center gap-2 mb-1">
								<span className="mono text-[10px] uppercase tracking-[0.18em] text-[#0369a1] font-bold">
									{c.id}
								</span>
								<span className="mono text-[9px] uppercase tracking-[0.15em] text-[#64748b]">atomic</span>
							</div>
							<p className="mono text-[13px] text-[#0b2447] leading-snug">{c.text}</p>
						</SchematicBox>
					</motion.div>
				))}
			</div>

			{/* Implementation path */}
			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 0.85 }}
				className="mt-6 w-full max-w-2xl"
			>
				<SchematicBox variant="ghost" className="px-4 py-2.5 rounded-sm">
					<div className="flex items-center justify-between gap-4">
						<BoxLabel>implementation</BoxLabel>
						<div className="flex items-center gap-3 mono text-[12px]">
							<span className="text-[#0b2447]" style={{ fontWeight: 600 }}>
								ollama-hosted LLM
							</span>
							<span className="text-[#64748b]">→ on failure →</span>
							<span className="text-[#0b2447]" style={{ fontWeight: 600 }}>
								rule-based fallback
							</span>
						</div>
					</div>
				</SchematicBox>
			</motion.div>
		</div>
	);
}
