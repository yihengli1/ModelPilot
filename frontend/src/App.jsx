import { useMemo, useState } from "react";

function App() {
	const [csvText, setCsvText] = useState("");
	const [rows, setRows] = useState([]);
	const [prompt, setPrompt] = useState("");
	const [error, setError] = useState("");

	const headers = useMemo(() => (rows.length ? rows[0] : []), [rows]);
	const bodyRows = useMemo(
		() => (rows.length > 1 ? rows.slice(1) : []),
		[rows]
	);

	const parseCsv = (text) => {
		const cleaned = text.trim();
		if (!cleaned) return [];
		return cleaned
			.split(/\r?\n/)
			.filter((line) => line.trim().length > 0)
			.map((line) => line.split(",").map((cell) => cell.trim()));
	};

	const handleParse = () => {
		try {
			const parsed = parseCsv(csvText);
			if (!parsed.length) {
				setError("Please paste a CSV with at least a header row.");
				setRows([]);
				return;
			}
			setRows(parsed);
			setError("");
		} catch (e) {
			setError("Could not parse CSV. Please check the format.");
			setRows([]);
		}
	};

	const handleFile = (file) => {
		if (!file) return;
		const reader = new FileReader();
		reader.onload = (event) => {
			setCsvText(event.target.result || "");
		};
		reader.readAsText(file);
	};

	return (
		<div className="min-h-screen bg-white text-slate-900">
			<div className="mx-auto max-w-5xl px-4 py-10">
				<header className="mb-8">
					<h1 className="text-3xl text-center font-semibold tracking-tight">
						ModelPilot Planner
					</h1>
					<p className="mt-2 text-slate-600">
						Paste or upload a CSV, preview it, and add optional context for the
						LLM.
					</p>
				</header>

				<section className="grid gap-6 lg:grid-cols-2">
					<div className="space-y-4">
						<div className="flex items-center gap-3">
							<label
								htmlFor="csv-upload"
								className="inline-flex cursor-pointer items-center gap-2 rounded border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
							>
								<input
									id="csv-upload"
									type="file"
									accept=".csv,text/csv"
									className="hidden"
									onChange={(e) => handleFile(e.target.files?.[0])}
								/>
								<span>Upload CSV</span>
							</label>
							<button
								type="button"
								onClick={handleParse}
								className="inline-flex items-center justify-center rounded bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
							>
								Insert CSV
							</button>
						</div>

						<textarea
							value={csvText}
							onChange={(e) => setCsvText(e.target.value)}
							placeholder="Paste CSV data here..."
							rows={12}
							className="w-full rounded border border-slate-300 p-3 text-sm font-mono text-slate-900 focus:border-slate-500 focus:outline-none"
						/>

						{error && <p className="text-sm text-rose-600">{error}</p>}
					</div>

					<div className="space-y-3">
						<div className="flex items-center justify-between">
							<h2 className="text-lg font-semibold">CSV Preview</h2>
							<span className="text-sm text-slate-500">
								{rows.length ? `${rows.length - 1} rows` : "No data yet"}
							</span>
						</div>
						<div className="overflow-auto rounded border border-slate-200">
							<table className="min-w-full border-collapse text-sm">
								<thead className="bg-slate-50">
									<tr>
										{headers.map((cell, idx) => (
											<th
												key={`${cell}-${idx}`}
												className="border border-slate-200 px-3 py-2 text-left font-semibold text-slate-700"
											>
												{cell || `Column ${idx + 1}`}
											</th>
										))}
										{!headers.length && (
											<th className="px-3 py-6 text-center text-slate-400">
												No CSV loaded
											</th>
										)}
									</tr>
								</thead>
								<tbody>
									{bodyRows.map((row, rIdx) => (
										<tr
											key={`row-${rIdx}`}
											className="odd:bg-white even:bg-slate-50"
										>
											{row.map((cell, cIdx) => (
												<td
													key={`cell-${rIdx}-${cIdx}`}
													className="border border-slate-200 px-3 py-2 text-slate-800"
												>
													{cell}
												</td>
											))}
										</tr>
									))}
									{!bodyRows.length && headers.length > 0 && (
										<tr>
											<td
												colSpan={headers.length}
												className="px-3 py-6 text-center text-slate-500"
											>
												Only header row detected. Add more data to preview rows.
											</td>
										</tr>
									)}
								</tbody>
							</table>
						</div>
					</div>
				</section>

				<div className="mt-8 space-y-3">
					<h2 className="text-lg font-semibold">LLM Context</h2>
					<p className="text-sm text-slate-600">
						Add any domain details, target definitions, or goals the model
						should consider.
					</p>
					<textarea
						value={prompt}
						onChange={(e) => setPrompt(e.target.value)}
						placeholder="Describe the dataset context, target column meaning, or what you want the model to optimize for..."
						rows={5}
						className="w-full rounded border border-slate-300 p-3 text-sm text-slate-900 focus:border-slate-500 focus:outline-none"
					/>
				</div>

				<div className="mt-8 space-y-3 flex">
					<button className="mx-auto flex-row bg-slate-900 text-white rounded text-lg h-12 w-60 hover:bg-slate-800">
						{" "}
						Generate Model
					</button>
				</div>
			</div>
		</div>
	);
}

export default App;
