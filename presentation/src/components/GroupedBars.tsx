import { motion } from 'framer-motion';

type Series = {
	label: string;
	values: number[];
	color: string;
	emphasised?: boolean;
};

type Props = {
	categories: string[];
	series: Series[];
	yMax?: number;
	yTicks?: number[];
	height?: number;
	yLabel?: string;
};

export function GroupedBars({
	categories,
	series,
	yMax = 1,
	yTicks = [0, 0.25, 0.5, 0.75, 1],
	height = 320,
	yLabel,
}: Props) {
	const groupCount = categories.length;
	const barsPerGroup = series.length;
	const groupGap = 36;
	const barGap = 6;
	const sidePad = 56;
	const viewWidth = 720;
	const innerWidth = viewWidth - sidePad * 2;
	const groupWidth = (innerWidth - groupGap * (groupCount - 1)) / groupCount;
	const barWidth = (groupWidth - barGap * (barsPerGroup - 1)) / barsPerGroup;

	return (
		<div className="w-full" style={{ height }}>
			<svg
				viewBox={`0 0 ${viewWidth} ${height}`}
				width="100%"
				height="100%"
				preserveAspectRatio="xMidYMid meet"
			>
				{/* Y axis grid + ticks */}
				{yTicks.map((t) => {
					const y = height - 32 - (t / yMax) * (height - 64);
					return (
						<g key={t}>
							<line
								x1={sidePad - 8}
								x2={viewWidth - sidePad}
								y1={y}
								y2={y}
								stroke="#cbd5e1"
								strokeWidth={0.5}
								strokeDasharray={t === 0 ? '0' : '2 3'}
							/>
							<text
								x={sidePad - 12}
								y={y + 3}
								textAnchor="end"
								className="mono"
								style={{ fontSize: 10, fill: '#64748b' }}
							>
								{t.toFixed(2)}
							</text>
						</g>
					);
				})}

				{/* Y axis label */}
				{yLabel && (
					<text
						x={14}
						y={height / 2}
						textAnchor="middle"
						transform={`rotate(-90 14 ${height / 2})`}
						className="mono"
						style={{ fontSize: 10, fill: '#64748b', letterSpacing: '0.18em', textTransform: 'uppercase' }}
					>
						{yLabel}
					</text>
				)}

				{/* Bars + category labels */}
				{categories.map((cat, i) => {
					const groupX = sidePad + i * (groupWidth + groupGap);
					return (
						<g key={cat}>
							{series.map((s, j) => {
								const value = s.values[i];
								const h = (value / yMax) * (height - 64);
								const x = groupX + j * (barWidth + barGap);
								const y = height - 32 - h;
								return (
									<g key={s.label}>
										<motion.rect
											initial={{ height: 0, y: height - 32 }}
											animate={{ height: h, y }}
											transition={{ delay: 0.15 + i * 0.08 + j * 0.04, duration: 0.5, ease: 'easeOut' }}
											x={x}
											width={barWidth}
											fill={s.color}
											stroke={s.emphasised ? '#0369a1' : 'none'}
											strokeWidth={s.emphasised ? 1.5 : 0}
										/>
										<motion.text
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											transition={{ delay: 0.6 + i * 0.08 + j * 0.04 }}
											x={x + barWidth / 2}
											y={y - 6}
											textAnchor="middle"
											className="mono"
											style={{
												fontSize: 11,
												fontWeight: 700,
												fill: s.emphasised ? '#0369a1' : '#0b2447',
											}}
										>
											{value.toFixed(3)}
										</motion.text>
									</g>
								);
							})}
							<text
								x={groupX + groupWidth / 2}
								y={height - 12}
								textAnchor="middle"
								className="mono"
								style={{ fontSize: 12, fill: '#0b2447', fontWeight: 600 }}
							>
								{cat}
							</text>
						</g>
					);
				})}
			</svg>

			{/* Legend */}
			<div className="flex justify-center gap-6 mt-2">
				{series.map((s) => (
					<div key={s.label} className="flex items-center gap-2">
						<span
							className="inline-block w-3 h-3"
							style={{ background: s.color, border: s.emphasised ? '1.5px solid #0369a1' : 'none' }}
						/>
						<span
							className="mono text-[11px] uppercase tracking-[0.12em]"
							style={{ color: '#0b2447', fontWeight: s.emphasised ? 700 : 500 }}
						>
							{s.label}
						</span>
					</div>
				))}
			</div>
		</div>
	);
}
