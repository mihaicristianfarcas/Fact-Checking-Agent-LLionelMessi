import type { ReactNode, CSSProperties } from 'react';
import { clsx } from 'clsx';

type Variant = 'solid' | 'dashed' | 'accent' | 'caveat' | 'ghost';

const VARIANT_STYLES: Record<Variant, CSSProperties> = {
	solid: { borderStyle: 'solid', borderColor: '#0b2447', borderWidth: 1, background: '#ffffff' },
	dashed: { borderStyle: 'dashed', borderColor: '#0b2447', borderWidth: 1, background: '#ffffff' },
	accent: { borderStyle: 'solid', borderColor: '#0ea5e9', borderWidth: 1.5, background: '#f0f9ff' },
	caveat: { borderStyle: 'solid', borderColor: '#d97706', borderWidth: 1.5, background: '#fffbeb' },
	ghost: { borderStyle: 'solid', borderColor: '#cbd5e1', borderWidth: 1, background: 'transparent' },
};

export function SchematicBox({
	children,
	variant = 'solid',
	corners = false,
	className,
	style,
}: {
	children: ReactNode;
	variant?: Variant;
	corners?: boolean;
	className?: string;
	style?: CSSProperties;
}) {
	return (
		<div
			className={clsx('relative', corners && 'corner-brackets', className)}
			style={{ ...VARIANT_STYLES[variant], ...style }}
		>
			{corners && (
				<>
					<span className="cb-bl" />
					<span className="cb-br" />
				</>
			)}
			{children}
		</div>
	);
}

export function BoxLabel({ children }: { children: ReactNode }) {
	return (
		<span className="mono text-[12px] uppercase tracking-[0.22em] text-[#0b2447]/60 font-semibold">
			{children}
		</span>
	);
}
