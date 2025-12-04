import csv
import io
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np

from openai import OpenAI


def _coerce_value(value):
    "string -> float"
    if value is None:
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return np.nan
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Value '{value}' cannot be converted to float.") from exc


def parse_csv_to_matrix(raw_csv: str) -> Tuple[List[str], List[dict], np.ndarray]:
    if not raw_csv.strip():
        raise ValueError("CSV content is empty.")

    try:
        sample = raw_csv[:1024]
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(raw_csv), dialect)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("CSV contains no data.")

    headers = rows[0]
    data_rows = rows[1:]

    parsed_rows = [
        {headers[i]: _coerce_value(row[i]) if i < len(
            row) else np.nan for i in range(len(headers))}
        for row in data_rows
    ]

    matrix_rows = [
        [_coerce_value(row[i]) if i < len(
            row) else np.nan for i in range(len(headers))]
        for row in data_rows
    ]
    matrix = np.array(matrix_rows, dtype=float)

    return headers, parsed_rows, matrix


def training_pipeline():

    # feature selection

    # Hyperparameter initialization 1 Call

    # Split Model, PyTorch training

    # Based on results 2 call

    # iterate over range of models/hyperparams/

    # best validation error

    # send back result

    pass


def generate_plan_from_gpt(
    *,
    system_context: str,
    prompt: str,
    dataset: np.ndarray,
):
    print(prompt)
    print(dataset)

    if isinstance(dataset, np.ndarray):
        if dataset.size == 0:
            raise ValueError("Dataset matrix cannot be empty.")
        dataset_for_prompt = dataset.tolist()
    else:
        raise ValueError("Dataset must a numpy matrix.")

    api_key = os.getenv("OPENAI_API_KEY")
    model_key = os.getenv("OPENAI_MODEL", "gpt-4o")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    user_message = f"""User prompt: {prompt or 'None provided.'}

CSV dataset:
{dataset_for_prompt}
""".strip()

    print(model_key)

    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_message},
    ]
    try:
        completion = client.chat.completions.create(
            model=model_key,
            messages=messages,
            temperature=1,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print(exc)
        raise

    print("TEST")
    content = completion.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw": content}

    print("content", parsed)

    return parsed
