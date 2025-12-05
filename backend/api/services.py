from openai import OpenAI
import numpy as np
import io
import json
import os
import pandas as pd
import random

from .contexts import TESTING_CONTEXT, TARGET_COLUMN_SYSTEM_CONTEXT


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


def generate_plan_gpt(
    *,
    prompt,
    summaries,
    target_name,

):
    api_key = os.getenv("OPENAI_API_KEY")
    model_key = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    user_message = f"""
###USER PROMPT:
{prompt or 'None provided.'}

###SUMMARY LIST:
{summaries}

###TARGET NAME:
{target_name}
""".strip()

    messages = [
        {"role": "system", "content": TESTING_CONTEXT},
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

    content = completion.choices[0].message.content
    parsed = json.loads(content)

    return parsed


def generate_target_gpt(
    *,
    user_prompt,
    headers,
):
    if not user_prompt:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    model_key = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    prompt_with_headers = f"""
    ### CANDIDATE COLUMNS
    {headers}

    ### USER INTENT
    {user_prompt}
    """

    messages = [
        {"role": "system", "content": TARGET_COLUMN_SYSTEM_CONTEXT},
        {"role": "user", "content": prompt_with_headers},
    ]

    try:
        completion = client.chat.completions.create(
            model=model_key,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
    except Exception as exc:
        print(exc)
        raise
    content = completion.choices[0].message.content

    target_column = content.strip().strip('"').strip("'")

    if target_column.upper() == "NONE" or target_column not in headers:
        return None

    return target_column


def summarize_and_select_features(
    headers,
    dataset_matrix,
    target_name
):
    if not headers or dataset_matrix.size == 0:
        raise NameError("Dataset or Headers is empty")
    df = pd.DataFrame(dataset_matrix, columns=headers)
    MAX_FEATURES_TO_SAMPLE = 50

    selected_summaries = []
    remaining_summaries = []

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

        if col_name == target_name:
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

    total_features = len(headers)

    aggregated_stats = {
        "dataset_shape": f"({df.shape[0]} rows, {df.shape[1]} columns)",
        "total_features": total_features,
        "features_sampled_for_summary": len(selected_summaries),
        "features_unsampled": unsampled_features_count,
        "overall_sparsity_ratio": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    }

    return selected_summaries, aggregated_stats
