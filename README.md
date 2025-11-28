# ModelPilot

LLM-assisted feature selection and model planning. Frontend uses React + Tailwind; backend is Django with a lightweight heuristic stub so you can drop in your own LLM + toolchain.

## Structure

- `backend/` Django project with API endpoints and an example dataset/prompt.
- `frontend/` React + Vite + Tailwind UI to paste data, send prompts, and display results.

## Backend quickstart

1. Create a virtualenv: `cd backend && python3 -m venv .venv && source .venv/bin/activate`.
2. Install deps: `pip install -r requirements.txt`.
3. Run migrations: `python manage.py migrate`.
4. Start dev server: `python manage.py runserver 0.0.0.0:8000`.
   - Sample endpoints: `GET /api/sample/`, `GET /api/runs/`, `POST /api/runs/create/`.

### API contract

`POST /api/runs/create/` expects JSON:

```json
{
	"dataset": "CSV text...",
	"prompt": "LLM instructions",
	"context": "optional extras",
	"target_column": "optional target"
}
```

Returns selected features, a suggested model, hyperparameters, and mock metrics. The logic lives in `backend/api/services.py` — replace it with real LLM/tool orchestration.

## Frontend quickstart

1. `cd frontend`.
2. Install deps: `npm install`.
3. Start dev server: `npm run dev` (defaults to port 5173).
4. The app calls `http://localhost:8000/api` by default; change `VITE_API_BASE` in `.env` or edit `src/lib/api.js`.

## Example data

- CSV: `backend/api/sample_data/example_dataset.csv`
- Prompt: `backend/api/sample_data/example_prompt.txt`

## Next steps

- Swap the heuristic stub with your LLM + tools (feature selection agent, hyperparameter search, model training).
- Add persistence for run artifacts, model files, and evaluation charts.
- Wire up an AWS or GPU runner to execute heavier training jobs asynchronously.
