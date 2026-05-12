import { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Radio } from 'lucide-react';
import { SLIDE_INFO } from './App';

const PresenterRemote = () => {
	const [currentSlide, setCurrentSlide] = useState(0);
	const [ws, setWs] = useState<WebSocket | null>(null);
	const [connected, setConnected] = useState(false);
	const [error, setError] = useState<string>('');

	useEffect(() => {
		let socket: WebSocket | null = null;
		let retryTimeout: ReturnType<typeof setTimeout>;

		const connect = () => {
			const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
			const wsUrl = `${protocol}//${window.location.host}/ws`;
			console.log('Connecting to:', wsUrl);

			socket = new WebSocket(wsUrl);

			socket.onopen = () => {
				console.log('WebSocket connected');
				setConnected(true);
				setError('');
				setWs(socket);
			};

			socket.onmessage = (event) => {
				try {
					const msg = JSON.parse(event.data);
					if (msg.type === 'sync') {
						setCurrentSlide(msg.slide);
					}
				} catch (e) {
					console.error('Parse error:', e);
				}
			};

			socket.onerror = (e) => {
				console.error('WebSocket error:', e);
				setError('Connection error');
			};

			socket.onclose = () => {
				console.log('WebSocket closed, retrying in 2s...');
				setConnected(false);
				setWs(null);
				// Retry connection after 2 seconds
				retryTimeout = setTimeout(connect, 2000);
			};
		};

		connect();

		return () => {
			clearTimeout(retryTimeout);
			socket?.close();
		};
	}, []);

	const sendCommand = useCallback((type: string) => {
		if (ws && ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify({ type }));
		}
	}, [ws]);

	const goToSlide = useCallback((index: number) => {
		if (ws && ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify({ type: 'goto', slide: index }));
		}
	}, [ws]);

	const nextSlide = () => sendCommand('next');
	const prevSlide = () => sendCommand('prev');

		return (
			<div className="h-screen bg-zinc-900 text-white p-6 flex flex-col overflow-hidden">
				{/* Header */}
				<div className="flex items-center justify-between mb-6 shrink-0">
					<h1 className="text-xl font-bold">Presenter Remote</h1>
					<div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${connected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
						<Radio size={14} className={connected ? '' : 'animate-pulse'} />
						{connected ? 'Connected' : 'Connecting...'}
					</div>
				</div>
				
				{error && (
					<div className="bg-red-500/20 text-red-400 px-4 py-2 rounded-lg mb-4 text-sm shrink-0">
						{error} - retrying...
					</div>
				)}
	
				{/* Current Slide Preview */}
				<div className="bg-zinc-800 rounded-2xl p-6 mb-6 shrink-0">
					<p className="text-zinc-400 text-sm mb-2">Current Slide</p>
					<p className="text-4xl font-black text-indigo-400">{currentSlide + 1} / {SLIDE_INFO.length}</p>
					<p className="text-xl font-medium mt-2">{SLIDE_INFO[currentSlide]?.name}</p>
				</div>
	
				{/* Navigation Buttons */}
				<div className="grid grid-cols-2 gap-4 mb-6 shrink-0">
					<button
						onClick={prevSlide}
						disabled={currentSlide === 0}
						className="bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 disabled:hover:bg-zinc-800 rounded-2xl p-8 flex flex-col items-center justify-center gap-3 transition-colors active:scale-95"
					>
						<ChevronLeft size={48} />
						<span className="text-lg font-medium">Previous</span>
					</button>
					<button
						onClick={nextSlide}
						disabled={currentSlide === SLIDE_INFO.length - 1}
						className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:hover:bg-indigo-600 rounded-2xl p-8 flex flex-col items-center justify-center gap-3 transition-colors active:scale-95"
					>
						<ChevronRight size={48} />
						<span className="text-lg font-medium">Next</span>
					</button>
				</div>
	
				{/* Slide List */}
				<div className="flex-1 overflow-auto min-h-0">
					<p className="text-zinc-400 text-sm mb-3">Jump to Slide</p>
					<div className="space-y-2">					{SLIDE_INFO.map((slide, i) => (
						<button
							key={i}
							onClick={() => goToSlide(i)}
							className={`w-full text-left px-4 py-3 rounded-xl transition-colors ${i === currentSlide
									? 'bg-indigo-600 text-white'
									: 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
								}`}
						>
							<span className="font-bold mr-3">{i + 1}.</span>
							{slide.name}
						</button>
					))}
				</div>
			</div>
		</div>
	);
};

export default PresenterRemote;
