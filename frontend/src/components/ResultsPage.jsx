import { useState, useMemo } from "react";
import { Link, useLocation, Navigate } from "react-router-dom";
import ResultsCard from "./ResultsCard";

const formatPercent = (val) => {
	if (val === undefined || val === null) return "N/A";
	return (val * 100).toFixed(2) + "%";
};

const formatKey = (key) => {
	return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
};

function formatNumberConditional(number, maxDecimalPlaces) {
	const numString = String(number);

	const decimalIndex = numString.indexOf(".");

	if (
		decimalIndex === -1 ||
		numString.length - 1 - decimalIndex <= maxDecimalPlaces
	) {
		return number;
	} else {
		return number.toFixed(maxDecimalPlaces);
	}
}

const formatRecursive = (value) => {
	if (Array.isArray(value)) {
		return `[${value.map(formatRecursive).join(", ")}]`;
	}
	return formatNumberConditional(value, 4);
};

function ResultsPage() {
	const { state } = useLocation();
	const [activeIndex, setActiveIndex] = useState(0);

	if (!state || !state.result) {
		return <Navigate to="/" />;
	}

	const { result, datasetText, dimensions } = state;

	const final_results = result.final_results;
	const plan = result.plan;
	const totalModels = final_results.length;

	const previewData = useMemo(() => {
		if (!datasetText) return { headers: [], rows: [] };

		const MAX_COLS = 20;
		const MAX_ROWS = 5;

		const lines = datasetText.trim().split(/\r?\n/);
		const nonEmptyLines = lines.filter((line) => line.trim().length > 0);

		if (nonEmptyLines.length === 0) return { headers: [], rows: [] };

		const headers = nonEmptyLines[0]
			.split(",")
			.map((c) => c.trim())
			.slice(0, MAX_COLS);
		const rows = nonEmptyLines.slice(1, MAX_ROWS + 1).map((line) =>
			line
				.split(",")
				.map((c) => c.trim())
				.slice(0, MAX_COLS)
		);
		return { headers, rows };
	}, [datasetText]);

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

	const currentModel = final_results[activeIndex];
	const currentSupervised = currentModel.metrics.supervised;

	const handlePrev = () => {
		setActiveIndex((prev) => (prev === 0 ? totalModels - 1 : prev - 1));
	};

	const handleNext = () => {
		setActiveIndex((prev) => (prev === totalModels - 1 ? 0 : prev + 1));
	};

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
								{plan.problem_type || "Unknown Type"}
							</span>
							<span>
								Target:{" "}
								<span className="font-semibold text-slate-900">
									{plan.target_column || "N/A"}
								</span>
							</span>
						</div>
					</div>
					<Link
						className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"
						to="/"
					>
						← New Dataset
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

						<div className="flex-1 overflow-hidden rounded-xl border border-slate-200 bg-white transition-all">
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
									{currentSupervised ? (
										<></>
									) : (
										<div className="text-center">
											<p className="text-xs font-bold uppercase tracking-wider text-slate-400">
												Unsupervised Model
											</p>
											<p className="text-xs font-bold uppercase tracking-wider text-slate-400">
												No validation/test acc
											</p>
										</div>
									)}
									<div className="text-center">
										<p className="text-xs font-bold uppercase tracking-wider text-slate-400">
											Validation Acc
										</p>
										<p className="font-mono text-2xl font-bold text-emerald-600">
											{currentSupervised
												? formatPercent(currentModel.metrics.val_accuracy)
												: formatPercent(0)}
										</p>
									</div>
									<div className="text-center">
										<p className="text-xs font-bold uppercase tracking-wider text-slate-400">
											Test Acc
										</p>
										<p className="font-mono text-2xl font-bold text-blue-600">
											{currentSupervised
												? formatPercent(currentModel.metrics.test_accuracy)
												: formatPercent(0)}
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
												const isLarge =
													Array.isArray(v) && (v.length > 5 || v[0].length > 5);

												return (
													<li
														key={k}
														className={`flex justify-between gap-4 ${
															isLarge ? "items-start" : "items-center"
														}`}
													>
														<span className="text-slate-600 font-medium shrink-0">
															{formatKey(k)}
														</span>
														<span
															className={`font-mono text-slate-800 ${
																isLarge
																	? "max-h-[5rem] overflow-y-auto w-2/3 text-right break-words bg-slate-50 p-2 rounded border border-slate-100 text-xs"
																	: ""
															}`}
														>
															{formatRecursive(v)}
														</span>
													</li>
												);
											})}
											{/* {Object.entries(currentModel.artifact).map(([k, v]) => {
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
															{formatRecursive(v)}
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
												)} */}
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
						<div className="overflow-x-auto rounded border border-slate-100 bg-slate-50">
							{previewData.rows.length > 0 ? (
								<table className="min-w-full border-collapse text-sm">
									<thead className="bg-slate-100">
										<tr>
											{previewData.headers.map((h, i) => (
												<th
													key={i}
													className="border border-slate-200 px-3 py-2 text-left font-semibold text-slate-700 whitespace-nowrap"
												>
													{h}
												</th>
											))}
										</tr>
									</thead>
									<tbody>
										{previewData.rows.map((row, rIdx) => (
											<tr
												key={rIdx}
												className="odd:bg-white even:bg-slate-50/50"
											>
												{row.map((cell, cIdx) => (
													<td
														key={cIdx}
														className="border border-slate-200 px-3 py-2 text-slate-700 whitespace-nowrap"
													>
														{cell}
													</td>
												))}
											</tr>
										))}
									</tbody>
								</table>
							) : (
								<p className="p-4 text-slate-500 italic">
									No dataset preview available.
								</p>
							)}
						</div>
					</div>

					<div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
						<h2 className="text-lg font-bold text-slate-900 mb-3">
							Pipeline Metadata
						</h2>
						<div className="space-y-3 text-sm">
							<ResultsCard data={dimensions?.totalRows} name="Total Rows" />
							<ResultsCard
								data={dimensions?.totalColumns}
								name="Total Columns"
							/>
							<ResultsCard data={plan.data_split?.method} name="Split Method" />
							<ResultsCard
								data={
									plan.data_split?.train_val_test
										? `[${plan.data_split.train_val_test.join(", ")}]`
										: "Default"
								}
								name="Ratios (Train/Val/Test)"
							/>

							<ResultsCard data={plan.total_tokens} name="Total tokens used" />
							<ResultsCard
								data={plan.total_models}
								name="Total models trained"
							/>
						</div>
					</div>
				</section>
			</div>
		</div>
	);
}

export default ResultsPage;
