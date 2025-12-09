import React from "react";

export default function ResultsCard({ data, name }) {
	return (
		<div className="flex justify-between border-b border-slate-100 pb-2">
			<span className="text-slate-500">{name}</span>
			<span className="font-medium text-slate-800">{data ?? "N/A"}</span>
		</div>
	);
}
