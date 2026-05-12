import { motion } from 'framer-motion';
import { AlertOctagon, Megaphone } from 'lucide-react';
import { SchematicBox, BoxLabel } from './SchematicBox';

export function ProblemSchematic() {
	const Panel = ({
		icon: Icon,
		label,
		title,
		desc,
		delay,
	}: {
		icon: typeof AlertOctagon;
		label: string;
		title: string;
		desc: string;
		delay: number;
	}) => (
		<motion.div
			initial={{ opacity: 0, y: 8 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ delay }}
			className="flex-1"
		>
			<SchematicBox variant="solid" className="p-6 rounded-sm h-full">
				<div className="flex items-center gap-3 mb-3">
					<Icon size={24} className="text-[#b91c1c]" strokeWidth={1.5} />
					<BoxLabel>{label}</BoxLabel>
				</div>
				<h3 className="text-2xl font-bold text-[#0b2447] mb-2 leading-tight">{title}</h3>
				<p className="text-[15px] text-[#64748b] leading-relaxed">{desc}</p>
			</SchematicBox>
		</motion.div>
	);

	return (
		<div className="flex flex-col gap-7 w-full">
			<div className="flex items-stretch gap-6">
				<Panel
					icon={Megaphone}
					label="failure mode 01"
					title="Over-aggressive classifiers"
					desc="Flag too eagerly. False positives bury real signal — humans stop trusting the system."
					delay={0.15}
				/>
				<Panel
					icon={AlertOctagon}
					label="failure mode 02"
					title="Confident LLMs"
					desc="Generate fluent verdicts that sound correct, cite sources that don't exist. Worse than useless — actively harmful."
					delay={0.3}
				/>
			</div>

			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 0.55 }}
				className="self-center flex items-center gap-4 px-6 py-3 bg-white"
				style={{ border: '1.5px solid #0ea5e9' }}
			>
				<span className="mono text-[10px] uppercase tracking-[0.22em] text-[#0369a1] font-bold">
					Design principle
				</span>
				<span className="w-px h-5 bg-[#0ea5e9]/40" />
				<span className="mono text-base text-[#0b2447] font-semibold tracking-tight">
					abstention &gt; fabrication
				</span>
			</motion.div>
		</div>
	);
}
