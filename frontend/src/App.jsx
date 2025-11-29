import InputPage from "./components/InputPage";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

function App() {
	return (
		<>
			<BrowserRouter>
				{/* Placeholder for now */}
				<Routes>
					<Route path="/" element={<InputPage />} />
				</Routes>
			</BrowserRouter>
		</>
	);
}

export default App;
