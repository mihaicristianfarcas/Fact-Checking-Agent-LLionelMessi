import { motion } from 'framer-motion';
import { Link as LinkIcon, Quote, ScrollText, Code2 } from 'lucide-react';

import { SlideShell, SlideHeader } from './components/SlideChrome';
import { SchematicBox, BoxLabel } from './components/SchematicBox';
import { MonoNumber } from './components/MonoNumber';
import { Caveat, OkBadge } from './components/Caveat';
import { GroupedBars } from './components/GroupedBars';
import { JourneyLine } from './components/JourneyLine';
import { Funnel } from './components/Funnel';
import { StatCard } from './components/StatCard';
import { PipelineDiagram } from './components/PipelineDiagram';
import { ProblemSchematic } from './components/ProblemSchematic';
import { ClaimProcessingDiagram } from './components/ClaimProcessingDiagram';
import { TrainingPipeline } from './components/TrainingPipeline';
import { OrchestratorDiagram } from './components/OrchestratorDiagram';
import { VerdictComparisonCard } from './components/VerdictComparisonCard';
import { StateOfTheArt } from './components/StateOfTheArt';

export type SlideDef = {
	id: string;
	name: string;
	render: () => React.ReactElement;
};

/* ─────────── 01 Title ─────────── */
const Title = () => (
	<SlideShell>
		<div className="flex-1 flex flex-col items-center justify-center text-center gap-12">
			<motion.div
				initial={{ opacity: 0, y: -8 }}
				animate={{ opacity: 1, y: 0 }}
				className="mono text-[11px] uppercase tracking-[0.32em] text-[#0ea5e9] font-semibold flex items-center gap-3"
			>
				<span className="w-8 h-px bg-[#0ea5e9]" />
				MAY 2026 · COURSE PRESENTATION
				<span className="w-8 h-px bg-[#0ea5e9]" />
			</motion.div>

			<div className="space-y-4">
				<motion.h1
					initial={{ opacity: 0, y: 12 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ delay: 0.1 }}
					className="text-[88px] leading-[0.95] font-black text-[#0b2447] tracking-tight"
				>
					Automated <br />
					<span style={{ color: '#0ea5e9' }}>Fact-Checking</span>
					<br /> Agent
				</motion.h1>
				<motion.p
					initial={{ opacity: 0, y: 8 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ delay: 0.2 }}
					className="mt-16 mono text-base text-[#64748b] uppercase tracking-[0.22em]"
				>
					tool-augmented RAG · trained verifier · 0% hallucinated citations
				</motion.p>
			</div>

			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 0.35 }}
				className="flex items-center gap-6"
			>
				{['Mihai-Cristian Farcaș', 'David Croitor', 'Andrei Ungureanu', 'Vasile Draguța'].map((name) => (
					<div
						key={name}
						className="px-4 py-2 mono text-sm text-[#0b2447] font-semibold tracking-tight hairline rounded-sm bg-white"
					>
						{name}
					</div>
				))}
			</motion.div>

			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 0.5 }}
				className="mono text-[11px] text-[#64748b]/70 uppercase tracking-[0.16em] flex flex-col items-center gap-1"
			>
				<span>team — LLionelMessi</span>
			</motion.div>
		</div>
	</SlideShell>
);

/* ─────────── 02 Problem ─────────── */
const Problem = () => (
	<SlideShell>
		<SlideHeader
			section="INTRO"
			figureId="02"
			figureTitle="FAILURE MODES"
			kicker="motivation"
			heading="Automated fact-checkers fail in two ways."
		/>
		<div className="flex-1 flex items-center justify-center pt-2">
			<ProblemSchematic />
		</div>
	</SlideShell>
);

/* ─────────── 03 State of the art ─────────── */
const SOTA = () => (
	<SlideShell>
		<SlideHeader
			section="INTRO"
			figureId="03"
			figureTitle="STATE OF THE ART"
			kicker="related work"
			heading="Existing systems either hallucinate citations or skip calibration."
		/>
		<div className="flex-1 flex items-center justify-center pt-2">
			<StateOfTheArt />
		</div>
	</SlideShell>
);

/* ─────────── 04 Architecture ─────────── */
const Architecture = () => (
	<SlideShell>
		<SlideHeader
			section="INTRO"
			figureId="04"
			figureTitle="AGENT PIPELINE"
			kicker="proposed solution"
			heading="Five tools. One orchestrator. Full audit trail."
		/>
		<div className="flex-1 flex flex-col items-center justify-center gap-6 pt-2">
			<PipelineDiagram />
			<p className="mono text-[11px] text-[#64748b] uppercase tracking-[0.16em] max-w-3xl text-center">
				Tool-augmented LLM pipeline. The orchestrator decides which tool to invoke at each step,
				accumulates evidence, and commits to a verdict — every citation traces back to a retrieved
				passage id.
			</p>
		</div>
	</SlideShell>
);

/* ─────────── 05 Corpus & filter ─────────── */
const Corpus = () => (
	<SlideShell>
		<SlideHeader
			section="A"
			figureId="05"
			figureTitle="CORPUS COMPOSITION"
			kicker="data & ingestion"
			heading="Filtering 5.4M wiki pages down to a political-claim corpus."
		/>
		<div className="flex-1 flex items-stretch gap-8 pt-4 min-h-0">
			{/* Left: funnel */}
			<div className="flex-1 flex flex-col justify-center">
				<Funnel
					stages={[
						{ label: 'FEVER wiki dump', value: '5,400,000', caption: 'pages', width: 100, tone: 'default' },
						{
							label: 'political-keyword filter',
							value: '263,000',
							caption: '~4.9% retained · politicians · institutions · elections',
							width: 70,
							tone: 'caveat',
						},
						{
							label: '~500-char chunks · MiniLM-L6-v2 embeddings',
							value: '662,806',
							caption: 'passages indexed in chromaDB',
							width: 48,
							tone: 'accent',
						},
					]}
				/>
			</div>

			{/* Right: stats + caveat */}
			<div className="w-[40%] flex flex-col gap-4 justify-center">
				<div className="grid grid-cols-2 gap-3">
					<StatCard label="claims · fever" value="145,449" />
					<StatCard label="claims · liar / politifact" value="12,836" />
					<StatCard label="total claims" value="158,285" emphasised />
					<StatCard label="sft triples" value="126K / 16K / 16K" caption="train / val / test" />
				</div>
				<Caveat>
					<strong>Selection bias is the trade-off.</strong> Indexing all 25M Wikipedia sentences was
					infeasible. We keep political pages only — claims outside that scope retrieve worse. The
					RFC scopes Phase 1 to political claims for exactly this reason.
				</Caveat>
			</div>
		</div>
	</SlideShell>
);

/* ─────────── 06 Retrieval breakthrough ─────────── */
const Retrieval = () => (
	<SlideShell>
		<SlideHeader
			section="A"
			figureId="06"
			figureTitle="RETRIEVAL RECALL"
			kicker="data & ingestion"
			heading="Dense embeddings alone weren't enough."
		/>
		<div className="flex-1 flex items-center gap-8 pt-4 min-h-0">
			<div className="flex-1">
				<GroupedBars
					categories={['k=1', 'k=5', 'k=10']}
					series={[
						{ label: 'dense-only', values: [0.015, 0.023, 0.023], color: '#94a3b8' },
						{
							label: 'hybrid · title index · reranker',
							values: [0.694, 0.848, 0.857],
							color: '#0ea5e9',
							emphasised: true,
						},
					]}
					yLabel="Recall @ k"
					height={340}
				/>
			</div>

			<div className="w-[34%] flex flex-col gap-4">
				<SchematicBox variant="accent" corners className="p-4 rounded-sm">
					<BoxLabel>headline</BoxLabel>
					<div className="mt-2 flex items-baseline gap-2 mono">
						<span className="text-base text-[#64748b]">Recall@10</span>
					</div>
					<div className="flex items-baseline gap-3 mt-1">
						<MonoNumber value="0.023" size="md" color="#64748b" />
						<span className="mono text-xl text-[#0ea5e9]">→</span>
						<MonoNumber value="0.857" size="md" color="#0369a1" weight={800} />
					</div>
					<p className="mono text-[10px] text-[#64748b] uppercase tracking-[0.14em] mt-2">
						on 500 FEVER dev claims
					</p>
				</SchematicBox>

				<SchematicBox variant="ghost" className="p-3 rounded-sm">
					<BoxLabel>configuration</BoxLabel>
					<ul className="mono text-[12px] text-[#0b2447] mt-2 space-y-1 leading-snug">
						<li>· FEVER title-lookup index (sqlite)</li>
						<li>· dense MiniLM-L6-v2 embeddings</li>
						<li>· cross-encoder reranker</li>
						<li className="mono text-[#64748b]">  ms-marco-MiniLM-L-6-v2</li>
						<li>· candidate_k = 50</li>
					</ul>
				</SchematicBox>
			</div>
		</div>
	</SlideShell>
);

/* ─────────── 07 Claim processing (decompose + stance) ─────────── */
const ClaimProcessing = () => (
	<SlideShell>
		<SlideHeader
			section="B"
			figureId="07"
			figureTitle="CLAIM PROCESSING"
			kicker="decompose · then stance"
			heading="Split compound claims, then label each (claim, passage) pair."
		/>
		<div className="flex-1 flex items-center justify-center pt-2">
			<ClaimProcessingDiagram />
		</div>
	</SlideShell>
);

/* ─────────── 08 SFT + DPO ─────────── */
const Training = () => (
	<SlideShell>
		<SlideHeader
			section="C"
			figureId="08"
			figureTitle="SFT + DPO · LoRA"
			kicker="model training"
			heading="Teach the format. Prefer well-cited answers."
		/>
		<div className="flex-1 flex flex-col gap-12 pt-10 min-h-0">
			<TrainingPipeline />

			<div className="grid grid-cols-3 gap-8 mt-1">
				<SchematicBox variant="solid" className="p-3 rounded-sm">
					<BoxLabel>base model</BoxLabel>
					<p className="mono text-[13px] text-[#0b2447] mt-1 font-semibold">TinyLlama-1.1B-Chat-v1.0</p>
				</SchematicBox>
				<SchematicBox variant="solid" className="p-3 rounded-sm">
					<BoxLabel>adapter</BoxLabel>
					<p className="mono text-[13px] text-[#0b2447] mt-1 font-semibold">LoRA</p>
					<p className="mono text-[10px] text-[#64748b] mt-0.5">r=8 · α=16</p>
				</SchematicBox>
				<SchematicBox variant="solid" className="p-3 rounded-sm">
					<BoxLabel>preference objective</BoxLabel>
					<p className="mono text-[12px] text-[#0b2447] mt-1 leading-snug">
						well-cited &gt; confident-unsupported
					</p>
				</SchematicBox>
			</div>

			<SchematicBox variant="ghost" className="px-4 py-3 rounded-sm flex items-center gap-3">
				<Quote size={16} className="text-[#0ea5e9] shrink-0" />
				<p className="mono text-[12px] text-[#0b2447] leading-snug">
					prompt: "cite passage ids explicitly · output{' '}
					<span className="text-[#0d9488] font-semibold">NOT_ENOUGH_INFO</span> when evidence is thin"
				</p>
			</SchematicBox>
		</div>
	</SlideShell>
);

/* ─────────── 09 Why two verdict heads ─────────── */
const Lessons = () => (
	<SlideShell>
		<SlideHeader
			section="C"
			figureId="09"
			figureTitle="VERDICT-HEAD EVALUATION"
			kicker="model training"
			heading="Two heads. The data picked the discriminative one."
		/>
		<div className="flex-1 flex flex-col gap-4 pt-3 min-h-0">
			<VerdictComparisonCard />

			<SchematicBox variant="ghost" className="px-4 py-3 rounded-sm">
				<BoxLabel>lesson</BoxLabel>
				<p className="text-[14px] text-[#0b2447] mt-1.5 leading-snug">
					<span className="font-semibold">Generative heads</span> produce fluent verdicts.{' '}
					<span className="font-semibold">Discriminative heads</span> produce calibrated 3-way labels.
					For this task, on this dataset, discriminative won.
				</p>
			</SchematicBox>
		</div>
	</SlideShell>
);

/* ─────────── 10 Synthesis & orchestrator ─────────── */
const Synthesis = () => (
	<SlideShell>
		<SlideHeader
			section="D"
			figureId="10"
			figureTitle="ORCHESTRATOR · TOOL CALLS"
			kicker="scoring · synthesis · eval"
			heading="The orchestrator wires it together — citations restricted by construction."
		/>
		<div className="flex-1 flex items-end gap-6 pb-16 min-h-0">
			<div className="flex-1">
				<OrchestratorDiagram />
			</div>

			<div className="w-[40%] flex flex-col gap-3 justify-center">
				<SchematicBox variant="accent" corners className="p-4 rounded-sm">
					<BoxLabel>citation tracing</BoxLabel>
					<p className="text-[13px] text-[#0b2447] mt-1.5 leading-snug">
						Cited passage IDs ⊆ retrieved passage IDs, enforced in the synthesizer. This is the
						mechanism behind 0.0% hallucinated citations.
					</p>
				</SchematicBox>

				<SchematicBox variant="solid" className="p-4 rounded-sm">
					<BoxLabel>credibility scorer</BoxLabel>
					<p className="mono text-[12px] text-[#0b2447] mt-1.5 leading-snug">
						source_relevance = f(title-match · rerank score · page-title penalty)
					</p>
					<p className="mono text-[11px] text-[#64748b] mt-2">
						default 0.5 when metadata absent — prevents weak-source guard from silently disengaging.
					</p>
				</SchematicBox>
			</div>
		</div>
	</SlideShell>
);

/* ─────────── 11 Trained verifier ─────────── */
const Verifier = () => (
	<SlideShell>
		<SlideHeader
			section="D"
			figureId="11"
			figureTitle="TRAINED VERIFIER"
			kicker="scoring · synthesis · eval"
			heading="Discriminative DeBERTa verifier, calibrated."
		/>
		<div className="flex-1 flex items-end gap-6 pb-16 min-h-0">
			<div className="flex-1 flex flex-col gap-3">
				<SchematicBox variant="solid" className="p-5 rounded-sm">
					<BoxLabel>training configuration</BoxLabel>
					<div className="grid grid-cols-2 gap-x-8 gap-y-2 mt-3 mono text-[12px] text-[#0b2447]">
						<div className="flex justify-between border-b border-[#e2e8f0] pb-1.5">
							<span className="text-[#64748b]">base</span>
							<span style={{ fontWeight: 700 }}>microsoft/deberta-v3-base</span>
						</div>
						<div className="flex justify-between border-b border-[#e2e8f0] pb-1.5">
							<span className="text-[#64748b]">dataset</span>
							<span style={{ fontWeight: 700 }}>20K FEVER train + 2K dev</span>
						</div>
						<div className="flex justify-between border-b border-[#e2e8f0] pb-1.5">
							<span className="text-[#64748b]">epochs</span>
							<span style={{ fontWeight: 700 }}>2 · fp16</span>
						</div>
						<div className="flex justify-between border-b border-[#e2e8f0] pb-1.5">
							<span className="text-[#64748b]">learning rate</span>
							<span style={{ fontWeight: 700 }}>2e-5</span>
						</div>
						<div className="flex justify-between border-b border-[#e2e8f0] pb-1.5">
							<span className="text-[#64748b]">class weighting</span>
							<span style={{ fontWeight: 700 }}>on</span>
						</div>
						<div className="flex justify-between border-b border-[#e2e8f0] pb-1.5">
							<span className="text-[#64748b]">max length</span>
							<span style={{ fontWeight: 700 }}>384</span>
						</div>
					</div>
				</SchematicBox>

				<SchematicBox variant="ghost" className="p-4 rounded-sm">
					<BoxLabel>calibration</BoxLabel>
					<p className="text-[13px] text-[#0b2447] mt-1.5 leading-snug">
						Temperature scalar T fit on dev split → written as sidecar{' '}
						<span className="mono">temperature.json</span>, loaded automatically.
					</p>
				</SchematicBox>
			</div>

			<div className="w-[40%] flex flex-col gap-3 justify-center">
				<BoxLabel>dev set · 2k claims</BoxLabel>
				<StatCard label="eval_accuracy" value="0.819" emphasised />
				<StatCard label="eval_macro_f1" value="0.795" emphasised />
				<SchematicBox variant="ghost" className="p-3 rounded-sm">
					<BoxLabel>hugging face</BoxLabel>
					<p className="mono text-[11px] text-[#0369a1] mt-1.5 leading-snug break-all">
						vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned
					</p>
				</SchematicBox>
			</div>
		</div>
	</SlideShell>
);

/* ─────────── 12 Headline result ─────────── */
const Result = () => (
	<SlideShell>
		<SlideHeader
			section="D"
			figureId="12"
			figureTitle="END-TO-END RESULT"
			kicker="scoring · synthesis · eval"
			heading="500-claim FEVER dev · decontaminated · held-out threshold tuning."
		/>
		<div className="flex-1 flex flex-col gap-6 pt-2 min-h-0">
			{/* Hero row */}
			<div className="flex items-stretch gap-6">
				<motion.div
					initial={{ opacity: 0, scale: 0.97 }}
					animate={{ opacity: 1, scale: 1 }}
					transition={{ delay: 0.1, duration: 0.4 }}
					className="flex-[1.4] hairline bg-white px-8 py-6 flex items-center justify-between"
				>
					<div className="flex flex-col gap-2">
						<BoxLabel>accuracy</BoxLabel>
						<MonoNumber value="0.704" size="hero" color="#0369a1" weight={800} />
						<div className="flex items-baseline gap-3">
							<span className="mono text-[11px] uppercase tracking-[0.18em] text-[#64748b]">
								macro F1
							</span>
							<MonoNumber value="0.701" size="md" color="#0b2447" weight={700} />
							<span className="mono text-[11px] uppercase tracking-[0.18em] text-[#64748b] ml-4">
								ECE
							</span>
							<MonoNumber value="0.185" size="md" color="#0b2447" weight={700} />
						</div>
					</div>
				</motion.div>

				<motion.div
					initial={{ opacity: 0, scale: 0.97 }}
					animate={{ opacity: 1, scale: 1 }}
					transition={{ delay: 0.25, duration: 0.4 }}
					className="flex-1 flex flex-col"
				>
					<SchematicBox variant="accent" corners className="p-6 rounded-sm h-full flex flex-col justify-center gap-2">
						<BoxLabel>hallucinated citations</BoxLabel>
						<div className="flex items-baseline gap-2">
							<MonoNumber value="0.0" size="xl" color="#0d9488" weight={800} />
							<span className="mono text-3xl text-[#0d9488]" style={{ fontWeight: 800 }}>
								%
							</span>
						</div>
						<OkBadge>Target was &lt; 5%</OkBadge>
						<p className="mono text-[11px] text-[#64748b] mt-1 leading-snug">
							citations restricted to retrieved passage ids by construction
						</p>
					</SchematicBox>
				</motion.div>
			</div>

			{/* Journey */}
			<div className="flex-1 hairline bg-white px-6 py-4 flex flex-col min-h-0">
				<div className="flex items-center justify-between mb-1">
					<BoxLabel>iteration journey · pipeline accuracy</BoxLabel>
					<span className="mono text-[10px] uppercase tracking-[0.14em] text-[#64748b]">
						baseline → final
					</span>
				</div>
				<div className="flex-1 min-h-0">
					<JourneyLine
						points={[
							{ value: 0.38, label: 'baseline RAG', sub: 'dense only' },
							{ value: 0.448, label: '+ title retrieval', sub: 'hybrid' },
							{ value: 0.51, label: '+ reranker', sub: 'cross-encoder' },
							{ value: 0.69, label: '+ source-aware synth.', sub: 'NLI head' },
							{ value: 0.704, label: '+ trained verifier', sub: 'calibrated' },
						]}
						yDomain={[0.3, 0.75]}
						height={130}
					/>
				</div>
			</div>
		</div>
	</SlideShell>
);

/* ─────────── 13 Demo video ─────────── */
const Demo = () => (
	<SlideShell>
		<SlideHeader
			section="DEMO"
			figureId="13"
			figureTitle="LIVE PIPELINE — SCREEN CAPTURE"
			kicker="demo"
			heading="Three political claims · three verdicts · zero hallucinated citations."
		/>
		<div className="flex-1 flex items-stretch gap-5 pt-3 min-h-0">
			<div className="flex-[1.7] hairline bg-[#0b2447] relative overflow-hidden">
				<video
					src="/demo.mp4"
					playsInline
					controls
					className="w-full h-full object-contain"
				/>
			</div>

			<div className="w-[36%] flex flex-col gap-3">
				<BoxLabel>three claims</BoxLabel>
				{[
					{ tag: 'SUPPORTED', color: '#0d9488', text: '"Barack Obama was the 44th President of the United States."' },
					{
						tag: 'REFUTED',
						color: '#b91c1c',
						text: '"The first president of Romania was Lionel Messi."',
					},
					{
						tag: 'NEI',
						color: '#d97706',
						text: '"The next Romanian president after 2025 will lead a center-right coalition."',
					},
				].map((c, i) => (
					<motion.div
						key={i}
						initial={{ opacity: 0, x: 8 }}
						animate={{ opacity: 1, x: 0 }}
						transition={{ delay: 0.15 + i * 0.1 }}
					>
						<SchematicBox variant="solid" className="px-3 py-2.5 rounded-sm">
							<span
								className="mono text-[10px] uppercase tracking-[0.18em] inline-block px-1.5 py-0.5"
								style={{
									color: c.color,
									background: 'transparent',
									border: `1px solid ${c.color}`,
									fontWeight: 700,
								}}
							>
								{c.tag}
							</span>
							<p className="mono text-[11.5px] text-[#0b2447] mt-1.5 leading-snug">{c.text}</p>
						</SchematicBox>
					</motion.div>
				))}
			</div>
		</div>
	</SlideShell>
);

/* ─────────── 14 Close ─────────── */
const Close = () => (
	<SlideShell>
		<SlideHeader
			section="END"
			figureId="14"
			figureTitle="REFERENCES · THANKS"
			kicker="thank you"
			heading="0.0% hallucination on political fact-checking, by construction."
		/>
		<div className="flex-1 flex flex-col gap-5 pt-4 min-h-0">
			<SchematicBox variant="accent" className="px-6 py-5 rounded-sm">
				<p className="text-xl text-[#0b2447] leading-snug">
					Tool-augmented RAG · trained discriminative verifier · citation tracing →{' '}
					<span style={{ color: '#0369a1', fontWeight: 700 }}>0.704 accuracy</span>,{' '}
					<span style={{ color: '#0d9488', fontWeight: 700 }}>0.0% hallucinated citations</span>{' '}
					on 500-claim FEVER decontaminated dev.
				</p>
			</SchematicBox>

			<div className="grid grid-cols-3 gap-3">
				<SchematicBox variant="solid" className="p-4 rounded-sm">
					<div className="flex items-center gap-2 mb-2">
						<Code2 size={14} className="text-[#0b2447]" />
						<BoxLabel>code</BoxLabel>
					</div>
					<p className="mono text-[11px] text-[#0369a1] break-all leading-snug">
						github.com/mihaicristianfarcas/Fact-Checking-Agent-LLionelMessi
					</p>
				</SchematicBox>
				<SchematicBox variant="solid" className="p-4 rounded-sm">
					<div className="flex items-center gap-2 mb-2">
						<LinkIcon size={14} className="text-[#0b2447]" />
						<BoxLabel>verifier · hugging face</BoxLabel>
					</div>
					<p className="mono text-[11px] text-[#0369a1] break-all leading-snug">
						vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned
					</p>
				</SchematicBox>
				<SchematicBox variant="solid" className="p-4 rounded-sm">
					<div className="flex items-center gap-2 mb-2">
						<LinkIcon size={14} className="text-[#0b2447]" />
						<BoxLabel>tinyllama adapter · hugging face</BoxLabel>
					</div>
					<p className="mono text-[11px] text-[#0369a1] break-all leading-snug">
						andreiungureanu/Fact-Checking-Agent-LLionelMessi
					</p>
				</SchematicBox>
			</div>

			<div className="flex items-center justify-center gap-4 mt-2">
				<span className="w-12 h-px bg-[#cbd5e1]" />
				<span className="mono text-[11px] uppercase tracking-[0.22em] text-[#0ea5e9] font-bold">
					thank you!
				</span>
				<span className="w-12 h-px bg-[#cbd5e1]" />
			</div>
		</div>
	</SlideShell>
);

export const SLIDES: SlideDef[] = [
	{ id: 'title', name: 'Title', render: Title },
	{ id: 'problem', name: 'The Problem', render: Problem },
	{ id: 'sota', name: 'State of the Art', render: SOTA },
	{ id: 'architecture', name: 'Architecture', render: Architecture },
	{ id: 'corpus', name: 'A · Corpus & Filter', render: Corpus },
	{ id: 'retrieval', name: 'A · Retrieval', render: Retrieval },
	{ id: 'claim-processing', name: 'B · Claim Processing', render: ClaimProcessing },
	{ id: 'training', name: 'C · SFT + DPO', render: Training },
	{ id: 'lessons', name: 'C · Verdict-Head Eval', render: Lessons },
	{ id: 'synthesis', name: 'D · Synthesis & Orchestrator', render: Synthesis },
	{ id: 'verifier', name: 'D · Trained Verifier', render: Verifier },
	{ id: 'result', name: 'D · Headline Result', render: Result },
	{ id: 'demo', name: 'Demo', render: Demo },
	{ id: 'close', name: 'Close', render: Close },
];

