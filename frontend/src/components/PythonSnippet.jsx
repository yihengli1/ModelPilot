import React from "react";
import { useMemo } from "react";

const pyValue = (v) => {
	if (v === null || v === undefined) return "None";
	if (typeof v === "string") return JSON.stringify(v); // adds quotes
	if (typeof v === "boolean") return v ? "True" : "False";
	if (Array.isArray(v)) return `[${v.map(pyValue).join(", ")}]`;
	return String(v);
};

const buildPythonSnippet = (modelName, params, targetColumn = null) => {
	const p = params || {};

	const MAP = {
		decision_tree: {
			import: "from sklearn.tree import DecisionTreeClassifier",
			cls: "DecisionTreeClassifier",
			extra: ", random_state=42",
			supervised: true,
		},
		knn: {
			import: "from sklearn.neighbors import KNeighborsClassifier",
			cls: "KNeighborsClassifier",
			extra: "",
			supervised: true,
		},
		naive_bayes: {
			import: "from sklearn.naive_bayes import GaussianNB",
			cls: "GaussianNB",
			extra: "",
			supervised: true,
		},
		kmeans: {
			import: "from sklearn.cluster import KMeans",
			cls: "KMeans",
			extra: ", random_state=42",
			supervised: false,
		},
		dbscan: {
			import: "from sklearn.cluster import DBSCAN",
			cls: "DBSCAN",
			extra: "",
			supervised: false,
		},
		hierarchical: {
			import: "from sklearn.cluster import AgglomerativeClustering",
			cls: "AgglomerativeClustering",
			extra: "",
			supervised: false,
		},
		linear_regression: {
			import:
				"from yourpkg.complexmodels.linear_regression import LinearRegressionTorchNN",
			cls: "LinearRegressionTorchNN",
			extra: "",
			supervised: true,
			custom: true,
		},
		kernel_polynomial: {
			import:
				"from yourpkg.complexmodels.kernel_polynomial import KernelPolynomialTorch",
			cls: "KernelPolynomialTorch",
			extra: "",
			supervised: true,
			custom: true,
		},
		linear_classifier: {
			import:
				"from yourpkg.complexmodels.linear_classifier import LinearClassifierTorchNN",
			cls: "LinearClassifierTorchNN",
			extra: "",
			supervised: true,
			custom: true,
		},
	};

	const info = MAP[modelName];
	if (!info) return `# No snippet template for model: ${modelName}`;

	const entries = Object.entries(p);
	const paramStr =
		entries.length > 0
			? entries.map(([k, v]) => `${k}=${pyValue(v)}`).join(", ")
			: "";

	const extraStr = info.extra ? info.extra.replace(/^,\s*/, "") : "";
	const fullArgs =
		paramStr && extraStr
			? `${paramStr}, ${extraStr}`
			: paramStr || extraStr || "";

	// For supervised models, we want a target column.
	const yLine = info.supervised
		? `target_col = ${
				targetColumn ? pyValue(targetColumn) : '"YOUR_TARGET_COLUMN"'
		  }
y = df[target_col].to_numpy()
X = df.drop(columns=[target_col]).to_numpy()`
		: `X = df.to_numpy()`;

	// Fit/predict differs for clustering
	const fitLines = info.supervised
		? `model.fit(X, y)
preds = model.predict(X)`
		: `labels = model.fit_predict(X)  # cluster labels`;

	return `import pandas as pd
import numpy as np

${info.import}

# 1) Load CSV
df = pd.read_csv("your_dataset.csv")

# 2) Convert to numpy arrays
${yLine}

# Note in ModelPilot we have already split the dataset
# into train/val/test sets, you can split through your own
# implementation

# 3) Build + train model
${
	info.custom == true
		? `
        # NOTE: This model is a custom class used by ModelPilot.
        # If you don't have the source code locally, use a sklearn model instead.`
		: "model = " + info.cls + "(" + fullArgs + ")"
}

${fitLines}
`;
};

export default function PythonSnippet({ currentModel }) {
	const usageSnippet = useMemo(() => {
		return buildPythonSnippet(currentModel.model, currentModel.hyperparameters);
	}, [currentModel.model, currentModel.hyperparameters]);

	return (
		<div className="pt-4 border-t border-slate-100 mx-8 mb-8">
			<h3 className="text-sm font-bold uppercase tracking-wide text-slate-400 pb-2">
				Use This Model (Python)
			</h3>

			<div className="relative rounded-lg border border-slate-800 bg-slate-950 shadow-sm">
				<pre
					className="text-xs leading-relaxed font-mono text-slate-100 p-4
                 max-h-64 overflow-x-auto overflow-y-auto overscroll-contain"
				>
					{usageSnippet}
				</pre>

				<button
					onClick={() => navigator.clipboard.writeText(usageSnippet)}
					className="absolute top-2 right-2 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 hover:bg-slate-800 active:bg-slate-700"
				>
					Copy
				</button>
			</div>
		</div>
	);
}
