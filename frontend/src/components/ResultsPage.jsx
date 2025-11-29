import React from "react";

const listifyObject = (obj) =>
	Object.entries(obj || {}).map(([key, value]) => ({
		key,
		value: typeof value === "object" ? JSON.stringify(value) : String(value),
	}));

function ResultsPage({ result, onBack }) {
	const payload = result;
	const modelName = payload.model;
	const hyperparameters = listifyObject(payload.hyperparameters);
	const metrics = payload.metrics;
	const selectedFeatures = payload.selected_features;

	const trainingError = metrics.training_error;
	const validationError = metrics.validation_error;

	const trainSplit = payload.training_split;
	const valSplit = payload.val_split;

	const datasetPreview = (result.datasetText || "")
		.trim()
		.split(/\r?\n/)
		.slice(0, 15)
		.join("\n");

	return (
		<div className="min-h-screen bg-main-white text-main-black">
			<div className="mx-auto max-w-5xl px-4 py-10">
				<header className="mb-8">
					<div className="flex items-center justify-between">
						<div>
							<h1 className="text-3xl font-semibold tracking-tight">
								Model Results
							</h1>
							<p className="mt-2 text-slate-600">
								Review the generated model details and dataset snapshot.
							</p>
						</div>
						<button
							onClick={onBack}
							className="rounded border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-main-white-hover"
						>
							Back to input
						</button>
					</div>
				</header>

				<section className="grid gap-4 sm:grid-cols-2">
					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Model</h2>
						<p className="text-sm text-slate-700">{modelName}</p>
						<p className="text-xs text-slate-500 mt-1">
							Features used: {selectedFeatures.length || "Not provided"}
						</p>
					</div>

					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Split</h2>
						<p className="text-sm text-slate-700">
							Training cut: {trainSplit} | Validation cut: {valSplit}
						</p>
						<p className="text-xs text-slate-500">
							Total rows: {result.dimensions?.totalRows ?? "N/A"} | Columns:{" "}
							{result.dimensions?.totalColumns ?? "N/A"}
						</p>
					</div>

					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Hyperparameters</h2>
						{hyperparameters.length ? (
							<ul className="space-y-1 text-sm text-slate-700">
								{hyperparameters.map(({ key, value }) => (
									<li key={key} className="flex justify-between gap-4">
										<span className="text-slate-600">{key}</span>
										<span className="font-medium">{value}</span>
									</li>
								))}
							</ul>
						) : (
							<p className="text-sm text-slate-600">Not provided</p>
						)}
					</div>

					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Errors</h2>
						<p className="text-sm text-slate-700">
							Training error: {trainingError}
						</p>
						<p className="text-sm text-slate-700">
							Validation error: {validationError}
						</p>
					</div>
				</section>

				<section className="mt-6 grid gap-4 sm:grid-cols-2">
					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Metrics</h2>
						{Object.keys(metrics).length ? (
							<ul className="space-y-1 text-sm text-slate-700">
								{Object.entries(metrics).map(([key, value]) => (
									<li key={key} className="flex justify-between gap-4">
										<span className="text-slate-600">{key}</span>
										<span className="font-medium">{String(value)}</span>
									</li>
								))}
							</ul>
						) : (
							<p className="text-sm text-slate-600">No metrics provided.</p>
						)}
					</div>

					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Selected Features</h2>
						{selectedFeatures.length ? (
							<ul className="flex flex-wrap gap-2 text-xs text-main-black">
								{selectedFeatures.map((feature) => (
									<li
										key={feature}
										className="rounded border border-slate-200 bg-main-white px-2 py-1"
									>
										{feature}
									</li>
								))}
							</ul>
						) : (
							<p className="text-sm text-slate-600">Not provided.</p>
						)}
					</div>

					<div className="rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Dataset Snapshot</h2>
						<div className="overflow-auto rounded border border-slate-200 bg-main-white p-2 text-xs text-main-black">
							<pre className="whitespace-pre-wrap break-words">
								{datasetPreview || "No dataset found."}
							</pre>
						</div>
					</div>
				</section>

				{payload.notes && (
					<section className="mt-6 rounded border border-slate-200 bg-main-white-hover p-4">
						<h2 className="text-lg font-semibold mb-2">Notes</h2>
						<p className="text-sm text-slate-700 whitespace-pre-wrap">
							{payload.notes}
						</p>
					</section>
				)}
			</div>
		</div>
	);
}

export default ResultsPage;
