import {
	BrowserRouter,
	Routes,
	Route,
	useLocation,
	useNavigate,
} from "react-router-dom";
import InputPage from "./components/InputPage";
import ResultsPage from "./components/ResultsPage";
import { fakeResultLarge, fakeResultSmall } from "./lib/testing";

const ResultsRoute = () => {
	const location = useLocation();
	const navigate = useNavigate();
	const { result, datasetText, dimensions } = location.state || {};

	return (
		<ResultsPage
			result={result || {}}
			datasetText={datasetText || ""}
			dimensions={dimensions || { totalRows: 0, totalColumns: 0 }}
			onBack={() => navigate("/")}
		/>
	);
};

function App() {
	return (
		<BrowserRouter>
			<Routes>
				<Route path="/" element={<InputPage />} />
				<Route path="/results" element={<ResultsRoute />} />
				<Route
					path="/testing1"
					element={
						<ResultsPage
							result={fakeResultSmall}
							onBack={() => navigate("/")}
						/>
					}
				/>
				<Route
					path="/testing2"
					element={
						<ResultsPage
							result={fakeResultLarge}
							onBack={() => navigate("/")}
						/>
					}
				/>
			</Routes>
		</BrowserRouter>
	);
}

export default App;
