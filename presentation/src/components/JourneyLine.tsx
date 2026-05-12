import { motion } from 'framer-motion';

type Point = { label: string; value: number; sub?: string };

export function JourneyLine({
	points,
	yDomain = [0.3, 0.75],
	height = 200,
}: {
	points: Point[];
	yDomain?: [number, number];
	height?: number;
}) {
	const w = 880;
	const padX = 110;
	const padTop = 22;
	const padBottom = 46;
	const innerW = w - padX * 2;
	const innerH = height - padTop - padBottom;
	const [yMin, yMax] = yDomain;
	const xs = points.map((_, i) => padX + (i * innerW) / (points.length - 1));
	const ys = points.map((p) => padTop + (1 - (p.value - yMin) / (yMax - yMin)) * innerH);
	const pathD = xs
		.map((x, i) => `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${ys[i].toFixed(1)}`)
		.join(' ');

	return (
		<svg viewBox={`0 0 ${w} ${height}`} width="100%" height={height}>
			{/* Y axis baseline + label */}
			<line x1={padX} x2={w - padX} y1={padTop + innerH} y2={padTop + innerH} stroke="#cbd5e1" strokeWidth={0.5} />

			{/* Animated journey line */}
			<motion.path
				d={pathD}
				stroke="#0ea5e9"
				strokeWidth={2.5}
				fill="none"
				strokeLinecap="round"
				strokeLinejoin="round"
				initial={{ pathLength: 0 }}
				animate={{ pathLength: 1 }}
				transition={{ duration: 1.6, ease: 'easeOut' }}
			/>

			{/* Points + labels */}
			{points.map((p, i) => (
				<g key={i}>
					<motion.circle
						cx={xs[i]}
						cy={ys[i]}
						r={i === points.length - 1 ? 6 : 4}
						fill={i === points.length - 1 ? '#0369a1' : '#0ea5e9'}
						stroke="#f5f7fb"
						strokeWidth={2}
						initial={{ scale: 0 }}
						animate={{ scale: 1 }}
						transition={{ delay: 0.15 + i * 0.18, duration: 0.3 }}
					/>
					<motion.text
						x={xs[i]}
						y={ys[i] - 12}
						textAnchor="middle"
						className="mono"
						style={{
							fontSize: 13,
							fontWeight: 700,
							fill: i === points.length - 1 ? '#0369a1' : '#0b2447',
						}}
						initial={{ opacity: 0, y: -4 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ delay: 0.3 + i * 0.18 }}
					>
						{p.value.toFixed(3)}
					</motion.text>
					<motion.text
						x={xs[i]}
						y={padTop + innerH + 14}
						textAnchor="middle"
						className="mono"
						style={{ fontSize: 10, fill: '#0b2447', fontWeight: 600, letterSpacing: '0.04em' }}
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						transition={{ delay: 0.4 + i * 0.18 }}
					>
						{p.label}
					</motion.text>
					{p.sub && (
						<motion.text
							x={xs[i]}
							y={padTop + innerH + 28}
							textAnchor="middle"
							className="mono"
							style={{ fontSize: 9, fill: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em' }}
							initial={{ opacity: 0 }}
							animate={{ opacity: 1 }}
							transition={{ delay: 0.45 + i * 0.18 }}
						>
							{p.sub}
						</motion.text>
					)}
				</g>
			))}
		</svg>
	);
}
