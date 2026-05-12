// Engineering Notebook color + type tokens.
// Restraint is the point — three hues, two fonts, no gradients except hero numbers.

export const color = {
	surface: '#f5f7fb',
	grid: '#dbe4f3',
	ink: '#0b2447',
	ink2: '#1e3a8a',
	muted: '#64748b',
	mutedSoft: '#94a3b8',
	hairline: '#cbd5e1',
	accent: '#0ea5e9',
	accentStrong: '#0369a1',
	accentSoft: '#e0f2fe',
	caveat: '#d97706',
	caveatSoft: '#fef3c7',
	ok: '#0d9488',
	okSoft: '#ccfbf1',
	refute: '#b91c1c',
	refuteSoft: '#fee2e2',
} as const;

export const font = {
	sans: "'Inter', system-ui, -apple-system, sans-serif",
	mono: "'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace",
} as const;

export type Color = keyof typeof color;
