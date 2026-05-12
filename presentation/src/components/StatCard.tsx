import { SchematicBox, BoxLabel } from './SchematicBox';
import { MonoNumber } from './MonoNumber';

export function StatCard({
	label,
	value,
	unit,
	caption,
	emphasised = false,
}: {
	label: string;
	value: string | number;
	unit?: string;
	caption?: string;
	emphasised?: boolean;
}) {
	return (
		<SchematicBox variant={emphasised ? 'accent' : 'solid'} className="px-5 py-4 rounded-sm flex flex-col gap-1">
			<BoxLabel>{label}</BoxLabel>
			<div className="flex items-baseline gap-1.5 mt-1">
				<MonoNumber value={value} size="md" color={emphasised ? '#0369a1' : '#0b2447'} weight={800} />
				{unit && (
					<span className="mono text-base text-[#64748b]" style={{ fontWeight: 600 }}>
						{unit}
					</span>
				)}
			</div>
			{caption && (
				<span className="mono text-[10px] uppercase tracking-[0.14em] text-[#64748b]">{caption}</span>
			)}
		</SchematicBox>
	);
}
