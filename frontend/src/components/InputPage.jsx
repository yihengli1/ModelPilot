import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { postCreate } from "../lib/services";

function InputPage() {
	const MAX_COLUMNS = 50000;
	const MAX_ROWS = 500000;
	const [rows, setRows] = useState([]);
	const [dimensions, setDimensions] = useState({
		totalRows: 0,
		totalColumns: 0,
	});
	const [fileName, setFileName] = useState("");
	const [fileRef, setFileRef] = useState(null);
	const [datasetText, setDatasetText] = useState("");
	const [prompt, setPrompt] = useState("");
	const [error, setError] = useState("");
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState("");
	const navigate = useNavigate();

	const headers = rows.length ? rows[0] : [];
	const bodyRows = useMemo(
		() => (rows.length > 1 ? rows.slice(1) : []),
		[rows]
	);

	const resetFileState = () => {
		setRows([]);
		setFileName("");
		setFileRef(null);
		setDatasetText("");
		setDimensions(() => ({
			totalRows: 0,
			totalColumns: 0,
		}));
	};

	const parseCsvPreview = (text) => {
		const cleaned = text.trim();
		if (!cleaned) {
			return { previewRows: [], totalRows: 0, totalColumns: 0 };
		}

		const lines = cleaned.split(/\r?\n/);
		const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
		const headerCells =
			nonEmptyLines.length > 0
				? nonEmptyLines[0].split(",").map((cell) => cell.trim())
				: [];
		const totalColumns = headerCells.length;
		const totalRows = Math.max(nonEmptyLines.length - 1, 0);

		const previewLines = nonEmptyLines.slice(0, 31); // header + 30 rows
		let maxCols = 0;

		const previewRows = previewLines.map((line) => {
			const cells = line.split(",").map((cell) => cell.trim());
			maxCols = Math.max(maxCols, cells.length);
			return cells.slice(0, 30); // limit to 30 columns
		});

		setDimensions(() => ({
			totalRows,
			totalColumns: Math.max(totalColumns, maxCols),
		}));

		return { previewRows, totalRows, totalColumns };
	};

	const handleFile = (file) => {
		if (!file) return;
		const reader = new FileReader();
		reader.onload = (event) => {
			const text = event.target.result || "";
			try {
				const { previewRows, totalRows, totalColumns } = parseCsvPreview(text);
				if (!previewRows.length) {
					setError("Uploaded CSV is empty or missing a header row.");
					resetFileState();
					return;
				}
				if (totalColumns > MAX_COLUMNS || totalRows > MAX_ROWS) {
					setError(
						`CSV too large. Limit is ${MAX_ROWS.toLocaleString()} rows and ${MAX_COLUMNS.toLocaleString()} columns.`
					);
					resetFileState();
					return;
				}
				setDatasetText(text);
				setRows(previewRows);
				setFileName(file.name);
				setFileRef(file);
				setError("");
			} catch (e) {
				setError("Could not parse CSV. Please check the format.");
				resetFileState();
			}
		};
		reader.readAsText(file);
	};

	const handleGenerate = async () => {
		if (!fileRef) {
			setSubmitError("Upload a CSV before generating a model.");
			return;
		}
		setSubmitError("");
		setSubmitting(true);
		try {
			const data = await postCreate(prompt, fileRef);

			console.log(data);
			// navigate("/results", {
			// 	state: {
			// 		result: data,
			// 		datasetText,
			// 		dimensions,
			// 	},
			// });
		} catch (err) {
			setSubmitError(err.message || "Failed to generate model.");
		} finally {
			setSubmitting(false);
		}
	};

	return (
		<div className="min-h-screen bg-main-white text-main-black">
			<div className="mx-auto max-w-5xl px-4 py-10">
				<header className="mb-8">
					<h1 className="text-3xl text-center font-semibold tracking-tight">
						ModelPilot
					</h1>
					<p className="mt-2 text-slate-600 text-center">
						Paste or upload a CSV, preview it, and add optional context for the
						LLM.
					</p>
				</header>

				<section className="flex gap-6 flex-col">
					<div className="space-y-4">
						<div className="flex items-center gap-3">
							<label
								htmlFor="csv-upload"
								className="inline-flex cursor-pointer items-center gap-2 rounded border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-main-white-hover mx-auto"
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
						</div>
						{fileName && (
							<p className="text-sm text-slate-600">
								Uploaded: {fileName}
								{fileRef?.size
									? ` (${(fileRef.size / 1024).toFixed(1)} KB)`
									: ""}
							</p>
						)}
						{rows.length > 0 ? (
							<p className="text-xs text-slate-500">
								Preview limited to first 30 columns and 30 data rows for
								performance.
							</p>
						) : (
							<></>
						)}

						{error && <p className="text-sm text-rose-600">{error}</p>}
					</div>

					<div className="space-y-3">
						<div className="flex items-center justify-between mt-[-1rem]">
							<h2 className="text-lg font-semibold">CSV Preview</h2>
							<div className="flex flex-col">
								<span className="text-sm text-slate-500">
									{rows.length ? `${dimensions.totalRows} rows` : "No data yet"}
								</span>
								<span className="text-sm text-slate-500">
									{rows.length ? `${dimensions.totalColumns} columns` : ""}
								</span>
							</div>
						</div>
						<div
							className={`overflow-auto rounded border border-slate-200 ${
								rows.length > 0 ? "h-80" : ""
							}`}
						>
							<table className="min-w-full border-collapse text-sm">
								<thead className="bg-main-white-hover">
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
											className="odd:bg-main-white even:bg-main-white-hover"
										>
											{row.map((cell, cIdx) => (
												<td
													key={`cell-${rIdx}-${cIdx}`}
													className="border border-slate-200 px-3 py-2 text-main-black-hover"
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
						className="w-full rounded border border-slate-300 p-3 text-sm text-main-black focus:border-slate-500 focus:outline-none"
					/>
				</div>

				<div className="mt-8 space-y-3 flex">
					<button
						className="mx-auto flex-row bg-main-black text-white rounded text-lg h-12 w-60 hover:bg-main-black-hover disabled:opacity-60 disabled:cursor-not-allowed"
						onClick={handleGenerate}
						disabled={submitting}
					>
						{submitting ? "Generating..." : "Generate Model"}
					</button>
				</div>

				{submitError && (
					<div className="mt-4 rounded border border-rose-600 bg-rose-50 p-3 text-center text-sm text-rose-700">
						{submitError}
					</div>
				)}
			</div>
		</div>
	);
}

export default InputPage;
