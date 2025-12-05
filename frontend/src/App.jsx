import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import InputPage from "./components/InputPage";
import ResultsPage from "./components/ResultsPage";
import testingData from "./lib/testingResult.json";

const { fakeResultLarge, fakeResultSmall } = testingData;

const ResultsRoute = () => {
	const location = useLocation();
	const { result, datasetText, dimensions } = location.state || {};

	return (
		<ResultsPage
			result={result || {}}
			datasetText={datasetText || ""}
			dimensions={dimensions || { totalRows: 0, totalColumns: 0 }}
		/>
	);
};

function App() {
	return (
		<BrowserRouter>
			<Routes>
				<Route path="/" element={<InputPage />} />
				<Route path="/results" element={<ResultsRoute />} />
				{/* <Route
					path="/testing1"
					element={<ResultsPage state={fakeResultSmall} />}
				/>
				<Route
					path="/testing2"
					element={<ResultsPage state={fakeResultLarge} />}
				/> */}
			</Routes>
		</BrowserRouter>
	);
}

export default App;
