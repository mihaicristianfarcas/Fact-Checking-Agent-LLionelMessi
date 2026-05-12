import type { ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

export function Caveat({ children }: { children: ReactNode }) {
	return (
		<div
			className="flex items-start gap-3 px-4 py-3 rounded-sm"
			style={{
				background: '#fffbeb',
				borderLeft: '3px solid #d97706',
			}}
		>
			<AlertTriangle size={18} className="text-[#d97706] shrink-0 mt-0.5" strokeWidth={2} />
			<div className="text-[13px] leading-snug text-[#854d0e]">{children}</div>
		</div>
	);
}

export function OkBadge({ children }: { children: ReactNode }) {
	return (
		<span
			className="mono inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm text-[11px] uppercase tracking-[0.15em] font-semibold"
			style={{ background: '#ccfbf1', color: '#0d9488', border: '1px solid #14b8a6' }}
		>
			<span className="w-1.5 h-1.5 rounded-full bg-[#0d9488]" />
			{children}
		</span>
	);
}
