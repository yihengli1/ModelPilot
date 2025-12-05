import { useState } from "react";
import { Link, useLocation, Navigate } from "react-router-dom";

const formatPercent = (val) => {
	if (val === undefined || val === null) return "N/A";
	return (val * 100).toFixed(2) + "%";
};

const formatKey = (key) => {
	return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
};

function ResultsPage() {
	const { state } = useLocation();
	const [activeIndex, setActiveIndex] = useState(0);

	if (!state || !state.result) {
		return <Navigate to="/" />;
	}

	const { result, datasetText, dimensions } = state;

	const initial_results = result.initial_results;
	const llm_results = result.llm_result;
	const totalModels = initial_results.length;

	if (totalModels === 0) {
		return (
			<div className="min-h-screen flex items-center justify-center">
				<p>No models were trained. Please check your inputs.</p>
				<Link to="/" className="ml-4 text-blue-600 underline">
					Go Back
				</Link>
			</div>
		);
	}

	const currentModel = initial_results[activeIndex];

	const handlePrev = () => {
		setActiveIndex((prev) => (prev === 0 ? totalModels - 1 : prev - 1));
	};

	const handleNext = () => {
		setActiveIndex((prev) => (prev === totalModels - 1 ? 0 : prev + 1));
	};

	// Dataset Preview Logic
	const datasetPreview = (datasetText || "")
		.trim()
		.split(/\r?\n/)
		.slice(0, 15)
		.join("\n");

	return (
		<div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
			<div className="mx-auto max-w-6xl px-4 py-10">
				<header className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
					<div>
						<h1 className="text-3xl font-bold tracking-tight text-slate-900">
							Model Results
						</h1>
						<div className="mt-2 flex items-center gap-4 text-sm text-slate-600">
							<span className="bg-slate-200 px-2 py-1 rounded text-xs font-semibold uppercase tracking-wider">
								{llm_results.problem_type || "Unknown Type"}
							</span>
							<span>
								Target:{" "}
								<span className="font-semibold text-slate-900">
									{llm_results.target_column || "N/A"}
								</span>
							</span>
						</div>
					</div>
					<Link
						className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"
						to="/"
					>
						← Start Over
					</Link>
				</header>

				<section className="relative mb-10">
					<div className="flex items-center gap-4">
						{totalModels > 1 ? (
							<button
								onClick={handlePrev}
								className="hidden md:flex h-12 w-12 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 hover:bg-slate-100 hover:text-slate-900 shadow-sm transition-all"
							>
								&larr;
							</button>
						) : (
							<></>
						)}

						<div className="flex-1 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg transition-all">
							<div className="border-b border-slate-100 bg-slate-50/50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
								<div>
									<div className="flex items-center gap-3">
										<h2 className="text-2xl font-bold capitalize text-slate-800">
											{currentModel.model?.replace(/_/g, " ")}
										</h2>
										<span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600">
											{activeIndex + 1} of {totalModels}
										</span>
									</div>
									{currentModel.error && (
										<span className="mt-1 inline-flex items-center rounded-md bg-red-50 px-2 py-1 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-600/10">
											Training Failed
										</span>
									)}
								</div>

								<div className="flex gap-8">
									<div className="text-center">
										<p className="text-xs font-bold uppercase tracking-wider text-slate-400">
											Validation Acc
										</p>
										<p className="font-mono text-2xl font-bold text-emerald-600">
											{formatPercent(currentModel.val_accuracy)}
										</p>
									</div>
									<div className="text-center">
										<p className="text-xs font-bold uppercase tracking-wider text-slate-400">
											Test Acc
										</p>
										<p className="font-mono text-2xl font-bold text-blue-600">
											{formatPercent(currentModel.test_accuracy)}
										</p>
									</div>
								</div>
							</div>

							<div className="p-6 grid gap-8 md:grid-cols-2">
								<div className="space-y-4">
									<h3 className="text-sm font-bold uppercase tracking-wide text-slate-400 border-b border-slate-100 pb-2">
										Hyperparameters
									</h3>
									{currentModel.hyperparameters &&
									Object.keys(currentModel.hyperparameters).length > 0 ? (
										<ul className="space-y-3 text-sm">
											{Object.entries(currentModel.hyperparameters).map(
												([k, v]) => (
													<li
														key={k}
														className="flex justify-between items-center"
													>
														<span className="text-slate-600 font-medium">
															{formatKey(k)}
														</span>
														<span className="font-mono bg-slate-100 px-2 py-1 rounded text-slate-800">
															{v === null ? "null" : String(v)}
														</span>
													</li>
												)
											)}
										</ul>
									) : (
										<p className="text-sm text-slate-500 italic">
											No hyperparameters tuned.
										</p>
									)}
								</div>

								<div className="space-y-4">
									<h3 className="text-sm font-bold uppercase tracking-wide text-slate-400 border-b border-slate-100 pb-2">
										Model Internals
									</h3>
									{currentModel.artifact && !currentModel.artifact.error ? (
										<ul className="space-y-3 text-sm">
											{Object.entries(currentModel.artifact).map(([k, v]) => {
												if (Array.isArray(v) && v.length > 5) return null; // Hide huge arrays
												return (
													<li
														key={k}
														className="flex justify-between items-center"
													>
														<span className="text-slate-600 font-medium">
															{formatKey(k)}
														</span>
														<span className="font-mono text-slate-800">
															{Array.isArray(v)
																? `[${v.join(", ")}]`
																: String(v)}
														</span>
													</li>
												);
											})}
											{currentModel.artifact.classes &&
												currentModel.artifact.classes.length > 5 && (
													<li className="flex justify-between items-center">
														<span className="text-slate-600 font-medium">
															Classes
														</span>
														<span className="font-mono text-slate-800">
															{currentModel.artifact.classes.length} unique
															labels
														</span>
													</li>
												)}
										</ul>
									) : (
										<p className="text-sm text-slate-500 italic">
											No artifacts available.
										</p>
									)}
								</div>
							</div>

							{currentModel.error && (
								<div className="bg-red-50 border-t border-red-100 p-4 text-sm text-red-800">
									<span className="font-bold">Error Details: </span>{" "}
									{currentModel.error}
								</div>
							)}
						</div>

						{totalModels > 1 ? (
							<button
								onClick={handleNext}
								className="hidden md:flex h-12 w-12 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 hover:bg-slate-100 hover:text-slate-900 shadow-sm transition-all"
							>
								&rarr;
							</button>
						) : (
							<></>
						)}
					</div>

					{totalModels > 1 ? (
						<div className="mt-4 flex justify-center gap-4 md:hidden">
							<button
								onClick={handlePrev}
								className="rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium hover:bg-slate-50"
							>
								Previous
							</button>
							<button
								onClick={handleNext}
								className="rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium hover:bg-slate-50"
							>
								Next
							</button>
						</div>
					) : (
						<></>
					)}
				</section>

				<section className="grid gap-6 md:grid-cols-2">
					<div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
						<h2 className="text-lg font-bold text-slate-900 mb-3">
							Dataset Snapshot
						</h2>
						<div className="overflow-x-auto rounded border border-slate-100 bg-slate-50 p-3">
							<pre className="text-xs text-slate-700 font-mono whitespace-pre-wrap break-words h-40 overflow-y-auto">
								{datasetPreview || "No dataset found."}
							</pre>
						</div>
					</div>

					<div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
						<h2 className="text-lg font-bold text-slate-900 mb-3">
							Pipeline Metadata
						</h2>
						<div className="space-y-3 text-sm">
							<div className="flex justify-between border-b border-slate-100 pb-2">
								<span className="text-slate-500">Total Rows</span>
								<span className="font-medium text-slate-800">
									{dimensions?.totalRows ?? "N/A"}
								</span>
							</div>
							<div className="flex justify-between border-b border-slate-100 pb-2">
								<span className="text-slate-500">Total Columns</span>
								<span className="font-medium text-slate-800">
									{dimensions?.totalColumns ?? "N/A"}
								</span>
							</div>
							<div className="flex justify-between border-b border-slate-100 pb-2">
								<span className="text-slate-500">Split Method</span>
								<span className="font-medium text-slate-800 capitalize">
									{llm_results.data_split?.method || "Random"}
								</span>
							</div>
							<div className="flex justify-between border-b border-slate-100 pb-2">
								<span className="text-slate-500">Ratios (Train/Val/Test)</span>
								<span className="font-medium text-slate-800">
									{llm_results.data_split?.train_val_test
										? `[${llm_results.data_split.train_val_test.join(", ")}]`
										: "Default"}
								</span>
							</div>
						</div>
					</div>
				</section>
			</div>
		</div>
	);
}

export default ResultsPage;
