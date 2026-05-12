import { motion } from 'framer-motion';
import {
	Scissors,
	Search,
	Scale,
	Layers,
	ShieldCheck,
	FileText,
	FileCheck,
} from 'lucide-react';

type Tool = {
	id: string;
	name: string;
	owner: 'A' | 'B' | 'C' | 'D';
	io: string;
	icon: typeof Scissors;
};

const TOOLS: Tool[] = [
	{ id: '01', name: 'Decomposer', owner: 'B', io: 'claim → atomic', icon: Scissors },
	{ id: '02', name: 'Retriever', owner: 'A', io: 'claim → passages', icon: Search },
	{ id: '03', name: 'Stance Cls.', owner: 'B', io: '(c, p) → S/R/N', icon: Scale },
	{ id: '04', name: 'Cred. Scorer', owner: 'D', io: 'source → trust', icon: ShieldCheck },
	{ id: '05', name: 'Verifier', owner: 'D', io: '→ verdict', icon: Layers },
];

const OWNER_COLOR: Record<Tool['owner'], string> = {
	A: '#0ea5e9',
	B: '#6366f1',
	C: '#a855f7',
	D: '#0d9488',
};

// Edge labels paired with each TOOL[i] (i.e. the arrow that points INTO tool i).
// First slot is the input → Decomposer arrow.
const EDGES = ['', 'top_k=5', 'candidate_k=50', 'max_passages=3', '+ calibration'];

const BOX_W = 100;
const EDGE_W = 68;

const BOX_BASE =
	'hairline bg-white px-2 py-2.5 flex flex-col items-center gap-1 relative shrink-0';

function EdgeArrow({
	label,
	delay,
}: {
	label?: string;
	delay: number;
}) {
	return (
		<div className="flex flex-col items-center justify-center shrink-0" style={{ width: EDGE_W }}>
			{label ? (
				<span
					className="mono text-[7.5px] text-[#0ea5e9] mb-1 whitespace-nowrap"
					style={{ fontWeight: 600, letterSpacing: '-0.01em' }}
				>
					{label}
				</span>
			) : (
				<span className="mono text-[7.5px] mb-1 opacity-0 select-none">·</span>
			)}
			<svg width={EDGE_W - 8} height="12" viewBox={`0 0 ${EDGE_W - 8} 12`}>
				<motion.line
					x1={0}
					x2={EDGE_W - 18}
					y1={6}
					y2={6}
					stroke="#0b2447"
					strokeWidth={1.2}
					initial={{ pathLength: 0 }}
					animate={{ pathLength: 1 }}
					transition={{ delay, duration: 0.3 }}
				/>
				<motion.polygon
					points={`${EDGE_W - 18},2 ${EDGE_W - 10},6 ${EDGE_W - 18},10`}
					fill="#0b2447"
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					transition={{ delay: delay + 0.25 }}
				/>
			</svg>
		</div>
	);
}

export function PipelineDiagram() {
	return (
		<div className="w-full flex flex-col items-center">
			<div className="flex items-center justify-center" style={{ gap: 4 }}>
				{/* Input box — same shape as tool box */}
				<motion.div
					initial={{ opacity: 0, y: 8 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ delay: 0.05 }}
					className={BOX_BASE}
					style={{ width: BOX_W }}
				>
					<span className="mono text-[9px] uppercase tracking-[0.1em] text-[#64748b]">io</span>
					<FileText size={18} className="text-[#0b2447]" strokeWidth={1.5} />
					<span className="text-[11.5px] font-bold text-[#0b2447] leading-tight text-center w-full">
						Input
					</span>
					<span
						className="mono text-[8.5px] text-[#64748b] text-center leading-tight w-full block"
						style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}
					>
						claim_text
					</span>
				</motion.div>

				{TOOLS.map((tool, i) => {
					const Icon = tool.icon;
					return (
						<div key={tool.id} className="flex items-center shrink-0" style={{ gap: 2 }}>
							<EdgeArrow label={EDGES[i]} delay={0.15 + i * 0.1} />

							<motion.div
								initial={{ opacity: 0, y: 8 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ delay: 0.1 + i * 0.1 }}
								className={BOX_BASE}
								style={{ width: BOX_W }}
							>
								{/* Owner pill */}
								<span
									className="absolute -top-2 right-1.5 mono text-[8px] uppercase tracking-[0.08em] px-1 py-0.5"
									style={{
										background: '#f5f7fb',
										border: `1px solid ${OWNER_COLOR[tool.owner]}`,
										color: OWNER_COLOR[tool.owner],
										fontWeight: 700,
									}}
								>
									§ {tool.owner}
								</span>
								<span className="mono text-[9px] uppercase tracking-[0.1em] text-[#64748b]">
									{tool.id}
								</span>
								<Icon size={18} className="text-[#0b2447]" strokeWidth={1.5} />
								<span className="text-[11.5px] font-bold text-[#0b2447] leading-tight text-center w-full">
									{tool.name}
								</span>
								<span
									className="mono text-[8.5px] text-[#64748b] text-center leading-tight w-full block"
									style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}
								>
									{tool.io}
								</span>
							</motion.div>
						</div>
					);
				})}

				{/* Output box — same shape, accent variant */}
				<div className="flex items-center shrink-0" style={{ gap: 2 }}>
					<EdgeArrow delay={0.7} />
					<motion.div
						initial={{ opacity: 0, y: 8 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ delay: 0.8 }}
						className="px-2 py-2.5 flex flex-col items-center gap-1 relative shrink-0"
						style={{
							width: BOX_W,
							background: '#e0f2fe',
							border: '1px solid #0ea5e9',
						}}
					>
						<span className="mono text-[9px] uppercase tracking-[0.1em] text-[#0369a1]">out</span>
						<FileCheck size={18} className="text-[#0369a1]" strokeWidth={1.5} />
						<span className="text-[11.5px] font-bold text-[#0369a1] leading-tight text-center w-full">
							Output
						</span>
						<span
							className="mono text-[8.5px] text-[#0369a1] text-center leading-tight w-full block"
							style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}
						>
							verdict + citations
						</span>
					</motion.div>
				</div>
			</div>

			{/* Owner legend */}
			<div className="mt-7 flex items-center justify-center gap-6 mono text-[10px] uppercase tracking-[0.14em] text-[#0b2447]/70">
				{Object.entries(OWNER_COLOR).map(([owner, c]) => (
					<div key={owner} className="flex items-center gap-1.5">
						<span className="w-2.5 h-2.5" style={{ background: c }} />
						<span style={{ fontWeight: 600 }}>§ {owner}</span>
					</div>
				))}
			</div>
		</div>
	);
}
