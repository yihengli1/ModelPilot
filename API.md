# API Documentation

## AutoML

### Run AutoML Prompt

Analyzes a dataset (provided as raw CSV text) based on a natural language prompt, generates a machine learning plan, trains multiple candidate models, and returns the best results.

**Endpoint:**
`POST /api/run-prompt/`

**Body Parameters:**

- `dataset` The raw string content of the CSV file.
- `prompt` Natural language description of the target variable or goal (e.g., "Predict the 'price' column").

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/run-prompt/" \
     -H "Content-Type: application/json" \
     -d '{
           "dataset": "age,income,churn\n25,50000,0\n30,60000,1",
           "prompt": "Predict churn using a decision tree"
         }'
```

**Response**

```json
{
	"prompt": "Predict churn using a decision tree",
	"dataset": "age,income,churn...",
	"plan": {
		"problem_type": "classification",
		"target_column": "churn",
		"recommended_models": [
			{
				"model": "decision_tree",
				"reasoning": "User requested specific model...",
				"initial_hyperparameters": { "max_depth": 5 }
			}
		],
		"data_split": { "method": "random", "train_val_test": [0.7, 0.15, 0.15] }
	},
	"final_results": [
		{
			"model": "decision_tree",
			"hyperparameters": { "max_depth": 5 },
			"val_accuracy": 0.85,
			"test_accuracy": 0.82,
			"artifact": { "depth": 5, "n_leaves": 8 }
		}
	]
}
```

### Datasets

**List Datasets**

Retrieve a list of all uploaded datasets.

**Endpoint:**

`GET /api/datasets/`

**Example Request:**

```bash
curl "http://localhost:8000/api/datasets/"
```

**Response**

```json
[
	{
		"id": 1,
		"file": "http://localhost:8000/media/datasets/data.csv",
		"uploaded_at": "2025-12-07T10:00:00Z"
	},
	{
		"id": 2,
		"file": "[https://my-bucket.s3.amazonaws.com/datasets/test.csv](https://my-bucket.s3.amazonaws.com/datasets/test.csv)",
		"uploaded_at": "2025-12-07T11:30:00Z"
	}
]
```

**Upload Datasets**

Upload a new dataset file (CSV) to the server.

**Endpoint:**

`POST /api/datasets/`

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/datasets/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/local/file.csv"
```

**Response**

```json
{
	"message": "File uploaded successfully",
	"id": 5,
	"url": "[https://my-bucket.s3.amazonaws.com/datasets/file.csv](https://my-bucket.s3.amazonaws.com/datasets/file.csv)"
}
```

**Get Dataset Info**

Retrieve metadata for a specific dataset by ID.

**Endpoint:**

`GET /api/datasets/:id/`

**Example Request:**

```bash
curl "http://localhost:8000/api/datasets/1/"
```

**Response**

```json
{
	"id": 1,
	"file": "http://localhost:8000/media/datasets/data.csv",
	"uploaded_at": "2025-12-07T10:00:00Z"
}
```
