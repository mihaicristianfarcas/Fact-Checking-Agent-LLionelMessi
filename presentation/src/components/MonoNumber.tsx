import { clsx } from 'clsx';

type Size = 'sm' | 'md' | 'lg' | 'xl' | 'hero';

const SIZE_CLASSES: Record<Size, string> = {
	sm: 'text-xl',
	md: 'text-4xl',
	lg: 'text-5xl',
	xl: 'text-7xl',
	hero: 'text-[6.5rem] leading-none',
};

export function MonoNumber({
	value,
	size = 'md',
	color = '#0b2447',
	weight = 700,
	className,
}: {
	value: string | number;
	size?: Size;
	color?: string;
	weight?: number;
	className?: string;
}) {
	return (
		<span
			className={clsx('mono tabular-nums', SIZE_CLASSES[size], className)}
			style={{ color, fontWeight: weight, letterSpacing: '-0.02em' }}
		>
			{value}
		</span>
	);
}
