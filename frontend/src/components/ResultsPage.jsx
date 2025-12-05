import React from "react";
import { Link, useLocation, Navigate } from "react-router-dom";

const formatPercent = (val) => {
	if (val === undefined || val === null) return "N/A";
	return (val * 100).toFixed(2) + "%";
};
const formatKey = (key) => key.replace(/_/g, " ");

function ResultsPage() {
	const { state } = useLocation();

	if (!state || !state.result) {
		return <Navigate to="/" />;
	}

	const { result, datasetText, dimensions } = state;
	const plan = result.plan || {};
	const modelResults = result.results || [];

	const datasetPreview = (datasetText || "")
		.trim()
		.split(/\r?\n/)
		.slice(0, 15)
		.join("\n");

	return (
		<div className="min-h-screen bg-main-white text-main-black">
			<div className="mx-auto max-w-6xl px-4 py-10">
				<header className="mb-8 flex items-center justify-between">
					<div>
						<h1 className="text-3xl font-semibold tracking-tight">
							Training Results
						</h1>
						<p className="mt-2 text-slate-600">
							Problem Type:{" "}
							<span className="font-medium text-main-black">
								{plan.problem_type || "Unknown"}
							</span>
							{" | "}
							Target:{" "}
							<span className="font-medium text-main-black">
								{plan.target_column || "None"}
							</span>
						</p>
					</div>
					<Link
						className="rounded border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-main-white-hover transition-colors"
						to="/"
					>
						← Start Over
					</Link>
				</header>
				<section className="space-y-8">
					{modelResults.map((modelData, index) => (
						<div
							key={index}
							className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm"
						>
							<div className="border-b border-slate-100 bg-slate-50/50 px-6 py-4 flex justify-between items-center">
								<div>
									<h2 className="text-xl font-bold capitalize text-slate-800">
										{modelData.model.replace(/_/g, " ")}
									</h2>
									{modelData.error && (
										<span className="inline-flex items-center rounded-md bg-red-50 px-2 py-1 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-600/10 mt-1">
											Training Failed
										</span>
									)}
								</div>
								<div className="flex gap-6 text-sm">
									<div className="text-center">
										<p className="text-xs text-slate-500 uppercase tracking-wider">
											Validation Acc
										</p>
										<p className="font-mono text-lg font-semibold text-emerald-600">
											{formatPercent(modelData.val_accuracy)}
										</p>
									</div>
									<div className="text-center">
										<p className="text-xs text-slate-500 uppercase tracking-wider">
											Test Acc
										</p>
										<p className="font-mono text-lg font-semibold text-sky-600">
											{formatPercent(modelData.test_accuracy)}
										</p>
									</div>
								</div>
							</div>

							<div className="p-6 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
								<div className="space-y-3">
									<h3 className="text-sm font-semibold text-slate-900 border-b border-slate-100 pb-2">
										Hyperparameters
									</h3>
									{modelData.hyperparameters &&
									Object.keys(modelData.hyperparameters).length > 0 ? (
										<ul className="space-y-2 text-sm">
											{Object.entries(modelData.hyperparameters).map(
												([k, v]) => (
													<li key={k} className="flex justify-between">
														<span className="text-slate-500 capitalize">
															{formatKey(k)}
														</span>
														<span className="font-mono text-slate-700">
															{String(v)}
														</span>
													</li>
												)
											)}
										</ul>
									) : (
										<p className="text-sm text-slate-400 italic">
											Defaults used
										</p>
									)}
								</div>

								<div className="space-y-3">
									<h3 className="text-sm font-semibold text-slate-900 border-b border-slate-100 pb-2">
										Model Artifacts
									</h3>
									{modelData.artifact && !modelData.artifact.error ? (
										<ul className="space-y-2 text-sm">
											{Object.entries(modelData.artifact).map(([k, v]) => {
												// Skip large arrays for UI cleanliness
												if (Array.isArray(v) && v.length > 5) return null;
												return (
													<li key={k} className="flex justify-between">
														<span className="text-slate-500 capitalize">
															{formatKey(k)}
														</span>
														<span className="font-mono text-slate-700">
															{Array.isArray(v) ? v.join(", ") : String(v)}
														</span>
													</li>
												);
											})}
											{modelData.artifact.classes &&
												modelData.artifact.classes.length > 5 && (
													<li className="flex justify-between">
														<span className="text-slate-500">Classes</span>
														<span className="font-mono text-slate-700">
															{modelData.artifact.classes.length} distinct
															classes
														</span>
													</li>
												)}
										</ul>
									) : (
										<p className="text-sm text-slate-400 italic">
											No artifacts available
										</p>
									)}
								</div>

								{modelData.error && (
									<div className="md:col-span-2 lg:col-span-1 rounded bg-red-50 p-3 text-xs text-red-700 border border-red-100">
										<strong>Error:</strong> {modelData.error}
									</div>
								)}
							</div>
						</div>
					))}
				</section>

				<section className="mt-8 grid gap-6 md:grid-cols-2">
					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Dataset Snapshot</h2>
						<div className="overflow-auto rounded border border-slate-200 bg-main-white p-2 text-xs text-main-black h-48">
							<pre className="whitespace-pre-wrap break-words font-mono">
								{datasetPreview || "No dataset found."}
							</pre>
						</div>
					</div>

					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-4">Metadata</h2>
						<div className="space-y-2 text-sm">
							<div className="flex justify-between border-b border-slate-200 pb-2">
								<span className="text-slate-600">Total Rows</span>
								<span className="font-medium">
									{dimensions?.totalRows ?? "N/A"}
								</span>
							</div>
							<div className="flex justify-between border-b border-slate-200 pb-2">
								<span className="text-slate-600">Total Columns</span>
								<span className="font-medium">
									{dimensions?.totalColumns ?? "N/A"}
								</span>
							</div>
							<div className="flex justify-between border-b border-slate-200 pb-2">
								<span className="text-slate-600">Split Method</span>
								<span className="font-medium capitalize">
									{plan.data_split?.method || "N/A"}
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
