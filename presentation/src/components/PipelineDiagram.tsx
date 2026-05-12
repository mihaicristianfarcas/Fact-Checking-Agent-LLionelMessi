import { motion } from 'framer-motion';
import { Scissors, Search, Scale, Layers, ShieldCheck } from 'lucide-react';

type Tool = {
	id: string;
	name: string;
	owner: 'A' | 'B' | 'C' | 'D';
	io: string;
	icon: typeof Scissors;
};

const TOOLS: Tool[] = [
	{ id: '01', name: 'Decomposer', owner: 'B', io: 'claim → atomic claims', icon: Scissors },
	{ id: '02', name: 'Retriever', owner: 'A', io: 'claim → passages', icon: Search },
	{ id: '03', name: 'Stance Cls.', owner: 'B', io: '(claim, p) → {S/R/N}', icon: Scale },
	{ id: '04', name: 'Cred. Scorer', owner: 'D', io: 'source → trust', icon: ShieldCheck },
	{ id: '05', name: 'Verifier', owner: 'D', io: 'signals → verdict', icon: Layers },
];

const OWNER_COLOR: Record<Tool['owner'], string> = {
	A: '#0ea5e9',
	B: '#6366f1',
	C: '#a855f7',
	D: '#0d9488',
};

const EDGES = ['top_k=5', 'candidate_k=50', 'max_passages=3', '+ calibration'];

export function PipelineDiagram() {
	return (
		<div className="w-full">
			<div className="flex items-stretch justify-between gap-3">
				{/* Input */}
				<motion.div
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					transition={{ delay: 0.05 }}
					className="flex flex-col items-center justify-center min-w-[110px]"
				>
					<div className="mono text-[10px] uppercase tracking-[0.18em] text-[#64748b] mb-2">input</div>
					<div className="px-3 py-2 hairline bg-white">
						<span className="mono text-xs text-[#0b2447]">claim_text</span>
					</div>
				</motion.div>

				{TOOLS.map((tool, i) => {
					const Icon = tool.icon;
					return (
						<div key={tool.id} className="flex items-center gap-2">
							{/* Edge w/ label */}
							<div className="flex flex-col items-center justify-center min-w-[60px]">
								<span className="mono text-[9px] uppercase tracking-[0.14em] text-[#0ea5e9] mb-1">
									{EDGES[i] ?? ''}
								</span>
								<svg width="60" height="14" viewBox="0 0 60 14">
									<motion.line
										x1={0}
										x2={48}
										y1={7}
										y2={7}
										stroke="#0b2447"
										strokeWidth={1.2}
										initial={{ pathLength: 0 }}
										animate={{ pathLength: 1 }}
										transition={{ delay: 0.15 + i * 0.1, duration: 0.3 }}
									/>
									<motion.polygon
										points="48,3 56,7 48,11"
										fill="#0b2447"
										initial={{ opacity: 0 }}
										animate={{ opacity: 1 }}
										transition={{ delay: 0.4 + i * 0.1 }}
									/>
								</svg>
							</div>

							<motion.div
								initial={{ opacity: 0, y: 8 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ delay: 0.1 + i * 0.1 }}
								className="hairline bg-white px-3 py-3 min-w-[130px] flex flex-col items-center gap-2 relative"
							>
								{/* Owner pill */}
								<span
									className="absolute -top-2 right-2 mono text-[9px] uppercase tracking-[0.12em] px-1.5 py-0.5"
									style={{
										background: '#f5f7fb',
										border: `1px solid ${OWNER_COLOR[tool.owner]}`,
										color: OWNER_COLOR[tool.owner],
										fontWeight: 700,
									}}
								>
									§ {tool.owner}
								</span>
								<span className="mono text-[10px] uppercase tracking-[0.12em] text-[#64748b]">{tool.id}</span>
								<Icon size={22} className="text-[#0b2447]" strokeWidth={1.5} />
								<span className="text-sm font-bold text-[#0b2447]">{tool.name}</span>
								<span className="mono text-[10px] text-[#64748b] text-center leading-tight">{tool.io}</span>
							</motion.div>
						</div>
					);
				})}

				{/* Output */}
				<div className="flex items-center gap-2">
					<svg width="60" height="14" viewBox="0 0 60 14">
						<motion.line
							x1={0}
							x2={48}
							y1={7}
							y2={7}
							stroke="#0b2447"
							strokeWidth={1.2}
							initial={{ pathLength: 0 }}
							animate={{ pathLength: 1 }}
							transition={{ delay: 0.7, duration: 0.3 }}
						/>
						<motion.polygon
							points="48,3 56,7 48,11"
							fill="#0b2447"
							initial={{ opacity: 0 }}
							animate={{ opacity: 1 }}
							transition={{ delay: 1 }}
						/>
					</svg>
					<motion.div
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						transition={{ delay: 0.8 }}
						className="flex flex-col items-center justify-center min-w-[120px]"
					>
						<div className="mono text-[10px] uppercase tracking-[0.18em] text-[#64748b] mb-2">output</div>
						<div className="px-3 py-2 bg-[#e0f2fe] border border-[#0ea5e9]">
							<span className="mono text-xs text-[#0369a1] font-semibold">verdict + citations</span>
						</div>
					</motion.div>
				</div>
			</div>

			{/* Owner legend */}
			<div className="mt-8 flex items-center justify-center gap-6 mono text-[10px] uppercase tracking-[0.14em] text-[#0b2447]/70">
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
