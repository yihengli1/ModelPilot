import csv
import io
from typing import List, Tuple
import numpy as np


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
        raise ValueError(f"Value '{value}' cannot be converted to float.") from exc


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
        {headers[i]: _coerce_value(row[i]) if i < len(row) else np.nan for i in range(len(headers))}
        for row in data_rows
    ]

    matrix_rows = [
        [_coerce_value(row[i]) if i < len(row) else np.nan for i in range(len(headers))]
        for row in data_rows
    ]
    matrix = np.array(matrix_rows, dtype=float)

    return headers, parsed_rows, matrix
