import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SLIDES } from './slides';

export const SLIDE_INFO = SLIDES.map((s) => ({ id: s.id, name: s.name }));
export const TOTAL_SLIDES = SLIDES.length;

const Presentation = () => {
	const [currentSlide, setCurrentSlide] = useState(() => {
		const saved = localStorage.getItem('presentation_slide');
		const initial = saved ? parseInt(saved, 10) : 0;
		return isNaN(initial) ? 0 : initial;
	});
	const [direction, setDirection] = useState(0);
	const [ws, setWs] = useState<WebSocket | null>(null);
	const [networkIP, setNetworkIP] = useState<string>('');

	useEffect(() => {
		localStorage.setItem('presentation_slide', currentSlide.toString());
	}, [currentSlide]);

	useEffect(() => {
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

		socket.onopen = () => setWs(socket);

		socket.onmessage = (event) => {
			try {
				const msg = JSON.parse(event.data);
				if (msg.type === 'next') {
					setDirection(1);
					setCurrentSlide((prev) => Math.min(prev + 1, TOTAL_SLIDES - 1));
				} else if (msg.type === 'prev') {
					setDirection(-1);
					setCurrentSlide((prev) => Math.max(prev - 1, 0));
				} else if (msg.type === 'sync') {
					const newSlide = Math.max(0, Math.min(msg.slide, TOTAL_SLIDES - 1));
					setCurrentSlide((prev) => {
						if (newSlide !== prev) {
							setDirection(newSlide > prev ? 1 : -1);
						}
						return newSlide;
					});
					if (msg.ip) setNetworkIP(msg.ip);
				}
			} catch (e) {
				console.error('WS parse error:', e);
			}
		};

		socket.onclose = () => setWs(null);
		return () => socket.close();
	}, []);

	useEffect(() => {
		if (ws && ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify({ type: 'goto', slide: currentSlide }));
		}
	}, [currentSlide, ws]);

	const nextSlide = useCallback(() => {
		if (currentSlide < SLIDES.length - 1) {
			setDirection(1);
			setCurrentSlide((prev) => prev + 1);
		}
	}, [currentSlide]);

	const prevSlide = useCallback(() => {
		if (currentSlide > 0) {
			setDirection(-1);
			setCurrentSlide((prev) => prev - 1);
		}
	}, [currentSlide]);

	useEffect(() => {
		const handleKeyDown = (e: KeyboardEvent) => {
			if (e.key === 'ArrowRight' || e.key === ' ') nextSlide();
			if (e.key === 'ArrowLeft') prevSlide();

			// Alt/Option + Digit 1-9 jumps to slide N. Use e.code for cross-platform.
			if (e.altKey && /^Digit[1-9]$/.test(e.code)) {
				e.preventDefault();
				const index = parseInt(e.code.replace('Digit', ''), 10) - 1;
				if (index < SLIDES.length && index !== currentSlide) {
					setDirection(index > currentSlide ? 1 : -1);
					setCurrentSlide(index);
				}
			}
		};
		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, [nextSlide, prevSlide, currentSlide]);

	const variants = {
		enter: (direction: number) => ({ x: direction > 0 ? 40 : -40, opacity: 0 }),
		center: { zIndex: 1, x: 0, opacity: 1 },
		exit: (direction: number) => ({ zIndex: 0, x: direction < 0 ? 40 : -40, opacity: 0 }),
	};

	const Slide = SLIDES[currentSlide].render;
	const isTitle = SLIDES[currentSlide].id === 'title';

	return (
		<div className="fixed inset-0 eng-grid flex flex-col items-center justify-center selection:bg-[#e0f2fe] selection:text-[#0369a1]">
			{/* Vignette wash */}
			<div className="absolute inset-0 pointer-events-none">
				<div
					className="absolute inset-0"
					style={{
						background:
							'radial-gradient(ellipse 90% 70% at 50% 50%, transparent 0%, rgba(245,247,251,0.6) 100%)',
					}}
				/>
			</div>

			{/* Top progress strip */}
			<div className="absolute top-0 left-0 w-full h-[3px] bg-[#dbe4f3] z-50">
				<motion.div
					initial={false}
					animate={{ width: `${((currentSlide + 1) / SLIDES.length) * 100}%` }}
					transition={{ duration: 0.25, ease: 'easeOut' }}
					className="h-full"
					style={{ background: '#0ea5e9' }}
				/>
			</div>

			{/* Top-left fixed deck label */}
			<div className="absolute top-4 left-6 z-40 flex items-center gap-3 mono text-[10px] uppercase tracking-[0.22em] text-[#0b2447]/60">
				<span className="w-1.5 h-1.5 rounded-full bg-[#0ea5e9]" />
				<span>fact-checking agent — llionelmessi</span>
			</div>

			{/* Top-right fixed timestamp */}
			<div className="absolute top-4 right-6 z-40 mono text-[10px] uppercase tracking-[0.18em] text-[#0b2447]/60">
				rfc · may 2026
			</div>

			{/* Main content */}
			<div className="relative w-full max-w-[1280px] aspect-[16/9] px-14 py-12 z-10">
				<AnimatePresence custom={direction} mode="wait">
					<motion.div
						key={currentSlide}
						custom={direction}
						variants={variants}
						initial="enter"
						animate="center"
						exit="exit"
						transition={{
							x: { type: 'tween', ease: 'easeOut', duration: 0.22 },
							opacity: { duration: 0.18 },
						}}
						className="w-full h-full flex flex-col"
					>
						<Slide />
					</motion.div>
				</AnimatePresence>
			</div>

			{/* Bottom chrome */}
			<div className="absolute bottom-5 left-0 w-full px-8 flex items-center justify-between z-20">
				<div className="flex items-center gap-6">
					<div className="mono text-[11px] text-[#0b2447]/80 tracking-[0.18em] uppercase flex items-center gap-2.5">
						<span style={{ fontWeight: 800 }}>{String(currentSlide + 1).padStart(2, '0')}</span>
						<span className="w-8 h-px bg-[#0b2447]/30" />
						<span className="text-[#0b2447]/40">{String(SLIDES.length).padStart(2, '0')}</span>
					</div>

					<div className="flex gap-1.5">
						{SLIDES.map((_, i) => (
							<button
								key={i}
								onClick={() => {
									setDirection(i > currentSlide ? 1 : -1);
									setCurrentSlide(i);
								}}
								className="h-1.5 transition-all duration-200"
								style={{
									width: i === currentSlide ? 22 : 6,
									background: i === currentSlide ? '#0ea5e9' : '#cbd5e1',
								}}
								aria-label={`Slide ${i + 1}`}
							/>
						))}
					</div>
				</div>

				{networkIP && isTitle && (
					<div className="mono text-[10px] text-[#0b2447]/30 tracking-[0.14em]">
						remote · {networkIP}:5173/present
					</div>
				)}
			</div>
		</div>
	);
};

export default Presentation;
