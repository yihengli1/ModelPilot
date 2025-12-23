const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const readFullFile = (file) =>
	new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = (event) => resolve(event.target?.result || "");
		reader.onerror = () => reject(new Error("Failed to read file"));
		reader.readAsText(file);
	});

export const postCreate = async (prompt, fileRef) => {
	const dataset = await readFullFile(fileRef);

	const resp = await fetch(`${API_BASE_URL}/run-prompt/`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			dataset,
			prompt,
		}),
	});
	if (!resp.ok) {
		const text = await resp.text();
		throw new Error(text || `Request failed with ${resp.status}`);
	}
	return await resp.json();
};

export const getAllDataset = async () => {
	const resp = await fetch(`${API_BASE_URL}/datasets/`, {
		method: "GET",
		headers: { "Content-Type": "application/json" },
	});
	if (!resp.ok) {
		const text = await resp.text();
		throw new Error(text || `Request failed with ${resp.status}`);
	}
	return await resp.json();
};

export const getDataset = async (id) => {
	console.log(`${API_BASE_URL}/datasets/${id}/`);
	const resp = await fetch(`${API_BASE_URL}/datasets/${id}/`, {
		method: "GET",
		headers: { "Content-Type": "application/json" },
	});

	if (!resp.ok) {
		const text = await resp.text();
		throw new Error(text || `Request failed with ${resp.status}`);
	}
	return await resp.json();
};

export const getExampleDataset = async () => {
	const resp = await fetch(`${API_BASE_URL}/datasets/examples/`, {
		method: "GET",
		headers: { "Content-Type": "application/json" },
	});

	if (!resp.ok) {
		const text = await resp.text();
		throw new Error(text || `Upload failed with ${resp.status}`);
	}

	return await resp.json();
};

export const wakeUpServer = async () => {
	try {
		await fetch(`${API_BASE_URL}/health/`, { method: "GET" });
	} catch (err) {
		console.error("Server wake-up failed (or is still booting):", err);
	}
};
