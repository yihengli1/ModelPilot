const API_BASE_URL = import.meta.env.VITE_API_BASE;

const readFullFile = (file) =>
	new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = (event) => resolve(event.target?.result || "");
		reader.onerror = () => reject(new Error("Failed to read file"));
		reader.readAsText(file);
	});

export const postCreate = async (prompt, fileRef) => {
	const dataset = await readFullFile(fileRef);

	const resp = await fetch(`${API_BASE_URL}/runs/create/`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			dataset,
			prompt,
			context: "",
			target_column: "",
		}),
	});
	if (!resp.ok) {
		const text = await resp.text();
		throw new Error(text || `Request failed with ${resp.status}`);
	}
	return await resp.json();
};
