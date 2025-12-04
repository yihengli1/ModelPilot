from openai import OpenAI
import numpy as np
import io
import json
import os
import pandas as pd


def parse_csv_to_matrix(raw_csv: str):
    if not raw_csv.strip():
        raise ValueError("CSV content is empty.")

    df = pd.read_csv(io.StringIO(raw_csv))

    headers = df.columns.tolist()
    parsed_rows = df.to_dict(orient="records")

    # numeric matrix (fallback to object if needed)
    try:
        matrix = df.to_numpy(dtype=float)
    except ValueError:
        matrix = df.to_numpy(dtype=object)

    return headers, parsed_rows, matrix


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
