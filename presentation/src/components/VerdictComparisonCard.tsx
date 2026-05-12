import { motion } from 'framer-motion';
import { SchematicBox, BoxLabel } from './SchematicBox';
import { MonoNumber } from './MonoNumber';

type Side = {
	id: string;
	model: string;
	sub: string;
	metrics: { label: string; value: string }[];
	verdictNote: string;
	winner: boolean;
};

const TINY: Side = {
	id: 'A',
	model: 'TinyLlama-1.1B',
	sub: 'LoRA SFT → DPO · generative head',
	metrics: [
		{ label: 'fluent verdicts', value: 'yes' },
		{ label: 'abstention smoke test', value: 'fail*' },
		{ label: 'citations restricted', value: 'guardrail' },
	],
	verdictNote: 'guardrail required — raw model not trusted',
	winner: false,
};

const DEBERTA: Side = {
	id: 'B',
	model: 'DeBERTa-v3-base',
	sub: '20K FEVER train · discriminative head',
	metrics: [
		{ label: 'eval_accuracy', value: '0.819' },
		{ label: 'eval_macro_f1', value: '0.795' },
		{ label: 'calibration', value: 'temperature' },
	],
	verdictNote: 'selected as verdict head',
	winner: true,
};

function SidePanel({ side, delay }: { side: Side; delay: number }) {
	return (
		<motion.div
			initial={{ opacity: 0, y: 8 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ delay }}
			className="flex-1"
		>
			<SchematicBox
				variant={side.winner ? 'accent' : 'solid'}
				className="p-5 rounded-sm h-full flex flex-col gap-3"
			>
				<div className="flex items-start justify-between">
					<BoxLabel>variant · {side.id}</BoxLabel>
					{side.winner && (
						<span className="mono text-[10px] uppercase tracking-[0.16em] text-[#0d9488] font-bold flex items-center gap-1">
							<span className="w-1.5 h-1.5 rounded-full bg-[#0d9488]" /> selected
						</span>
					)}
				</div>

				<div>
					<h4 className="text-xl font-bold text-[#0b2447] tracking-tight">{side.model}</h4>
					<p className="mono text-[11px] text-[#64748b] uppercase tracking-[0.12em] mt-0.5">{side.sub}</p>
				</div>

				<div className="border-t border-[#cbd5e1] pt-3 flex flex-col gap-1.5">
					{side.metrics.map((m) => (
						<div key={m.label} className="flex items-baseline justify-between">
							<span className="mono text-[11px] uppercase tracking-[0.12em] text-[#64748b]">{m.label}</span>
							<MonoNumber value={m.value} size="sm" color={side.winner ? '#0369a1' : '#0b2447'} weight={700} />
						</div>
					))}
				</div>

				<p className="mono text-[11px] text-[#0b2447] italic mt-auto">→ {side.verdictNote}</p>
			</SchematicBox>
		</motion.div>
	);
}

export function VerdictComparisonCard() {
	return (
		<div className="w-full">
			<div className="flex items-stretch gap-5">
				<SidePanel side={TINY} delay={0.15} />
				<SidePanel side={DEBERTA} delay={0.3} />
			</div>
			<p className="mono text-[11px] text-[#64748b] mt-3 italic">
				* Everest claim with Mariana Trench evidence — raw model returned SUPPORTED. We added a citation/evidence guardrail.
			</p>
		</div>
	);
}
