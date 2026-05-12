import { motion } from 'framer-motion';

type Stage = {
	label: string;
	value: string;
	caption: string;
	width: number;
	tone?: 'default' | 'accent' | 'caveat';
};

export function Funnel({ stages }: { stages: Stage[] }) {
	return (
		<div className="flex flex-col items-center gap-2 w-full">
			{stages.map((s, i) => {
				const toneStyles =
					s.tone === 'accent'
						? { background: '#e0f2fe', borderColor: '#0ea5e9', valueColor: '#0369a1' }
						: s.tone === 'caveat'
							? { background: '#fffbeb', borderColor: '#d97706', valueColor: '#92400e' }
							: { background: '#ffffff', borderColor: '#0b2447', valueColor: '#0b2447' };

				return (
					<div key={s.label} className="w-full flex flex-col items-center">
						<motion.div
							initial={{ opacity: 0, y: -10, width: '40%' }}
							animate={{ opacity: 1, y: 0, width: `${s.width}%` }}
							transition={{ delay: 0.15 + i * 0.18, duration: 0.45, ease: 'easeOut' }}
							className="flex items-center justify-between px-6 py-3"
							style={{
								background: toneStyles.background,
								border: `1px solid ${toneStyles.borderColor}`,
								borderRadius: 2,
							}}
						>
							<span
								className="mono text-[10px] uppercase tracking-[0.18em]"
								style={{ color: '#0b2447', fontWeight: 600, opacity: 0.7 }}
							>
								{s.label}
							</span>
							<span
								className="mono tabular-nums"
								style={{ fontSize: 22, fontWeight: 800, color: toneStyles.valueColor, letterSpacing: '-0.02em' }}
							>
								{s.value}
							</span>
						</motion.div>
						<motion.span
							initial={{ opacity: 0 }}
							animate={{ opacity: 1 }}
							transition={{ delay: 0.35 + i * 0.18 }}
							className="mono text-[10px] uppercase whitespace-nowrap tracking-[0.16em] text-[#64748b] mt-1"
						>
							{s.caption}
						</motion.span>
						{i < stages.length - 1 && (
							<motion.div
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								transition={{ delay: 0.45 + i * 0.18 }}
								className="mono text-[#0b2447] text-sm mt-2 mb-1"
								style={{ fontWeight: 700 }}
							>
								↓
							</motion.div>
						)}
					</div>
				);
			})}
		</div>
	);
}
