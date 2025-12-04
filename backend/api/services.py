from openai import OpenAI
import numpy as np
import io
import json
import os
import pandas as pd
import random


def parse_csv_to_matrix(raw_csv: str):
    if not raw_csv.strip():
        raise ValueError("CSV content is empty.")

    df = pd.read_csv(io.StringIO(raw_csv))

    # drop non numeric for now...
    def is_numeric_series(s):
        try:
            pd.to_numeric(s.dropna(), errors="raise")
            return True
        except Exception:
            return False

    numeric_cols = [col for col in df.columns if is_numeric_series(df[col])]
    df = df[numeric_cols]

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


def summarize_and_select_features(
    headers,
    dataset_matrix,
    target_name
):
    if not headers or not dataset_matrix:
        return [], {}
    df = pd.DataFrame(dataset_matrix, columns=headers)
    all_summaries = []

    MAX_FEATURES_TO_SAMPLE = 100

    for col_name in headers:
        series = df[col_name]

        inferred_dtype = str(series.dtype)
        # should always be numeric rn
        if inferred_dtype in ['int64', 'float64']:
            inferred_type = 'numeric'
        elif series.nunique() <= 50 and series.nunique() < len(series) * 0.1:
            inferred_type = 'categorical'
        else:
            inferred_type = 'text_id'

        summary = {
            "name": col_name,
            "role": "Target" if col_name == target_name else "Feature",
            "inferred_type": inferred_type,
            "total_count": len(series),
            "missing_ratio": series.isnull().sum() / len(series)
        }

        if inferred_type == 'numeric':
            summary["mean"] = series.mean()
            summary["std"] = series.std()
            summary["min"] = series.min()
            summary["max"] = series.max()
        elif inferred_type == 'categorical':
            summary["top_5_values"] = series.value_counts().nlargest(
                5).index.tolist()

        all_summaries.append(summary)

    selected_summaries = []
    remaining_summaries = []

    for summary in all_summaries:
        if summary['name'] == target_name:
            selected_summaries.append(summary)
        else:
            remaining_summaries.append(summary)

    slots_left = MAX_FEATURES_TO_SAMPLE - len(selected_summaries)

    if len(remaining_summaries) > slots_left:
        sampled_features = random.sample(remaining_summaries, slots_left)
        selected_summaries.extend(sampled_features)
        unsampled_features_count = len(remaining_summaries) - slots_left
    else:
        selected_summaries.extend(remaining_summaries)
        unsampled_features_count = 0

    # Calculate overall stats for context
    total_features = len(headers)
    numeric_count = df.select_dtypes(include=['number']).shape[1]
    categorical_count = df.select_dtypes(
        include=['object', 'category']).shape[1]

    aggregated_stats = {
        "dataset_shape": f"({df.shape[0]} rows, {df.shape[1]} columns)",
        "total_features": total_features,
        "features_sampled_for_summary": len(selected_summaries),
        "features_unsampled": unsampled_features_count,
        "type_breakdown": {
            "numeric": numeric_count,
            "categorical_or_text": categorical_count
        },
        "overall_sparsity_ratio": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    }

    return selected_summaries, aggregated_stats
