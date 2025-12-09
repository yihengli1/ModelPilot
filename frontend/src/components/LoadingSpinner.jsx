export default function LoadingSpinner() {
	return (
		<div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white/90 backdrop-blur-sm transition-all">
			<div className="relative flex items-center justify-center">
				<div className="absolute h-24 w-24 rounded-full border-4 border-slate-100 animate-ping opacity-75"></div>
				<div className="h-16 w-16 animate-spin rounded-full border-4 border-slate-200 border-t-slate-800"></div>
			</div>
			<h2 className="mt-8 text-2xl font-semibold text-slate-800 tracking-tight animate-pulse">
				Training Model
			</h2>
			<p className="mt-2 text-slate-500">
				Analyzing your data and generating a pipeline...
			</p>
			<p className="mt-2 text-slate-400">May take up to 1 minute...</p>
		</div>
	);
}
