import csv
import io
from typing import List, Tuple

import numpy as np
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from services import _coerce_value, parse_csv_to_matrix


class CreateRunView(APIView):
    """Handle the POST from the frontend to parse a CSV and emit a numpy matrix."""

    def post(self, request):
        dataset = request.data.get("dataset", "")
        prompt = request.data.get("prompt", "")
        context = request.data.get("context", "")
        target_column = request.data.get("target_column", "")

        try:
            headers, rows, matrix = parse_csv_to_matrix(dataset)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            return Response(
                {"error": "Failed to parse CSV content."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        response_payload = {
            "prompt": prompt,
            "context": context,
            "target_column": target_column,
            "headers": headers,
            "rows": rows,
            "matrix": matrix.tolist(),
            "shape": list(matrix.shape),
            "model": "csv-parser",
            "hyperparameters": {},
            "metrics": {
                "training_error": None,
                "validation_error": None,
            },
            "selected_features": headers,
            "training_split": None,
            "val_split": None,
            "optimizer": "n/a",
            "notes": "Dataset parsed into a numpy matrix; no model training performed.",
        }

        return Response(response_payload, status=status.HTTP_200_OK)


class RunViewSet(viewsets.ViewSet):
    """
    Minimal viewset placeholder for router compatibility.
    Delegates creation to CreateRunView for consistent behavior.
    """

    def list(self, request):
        return Response([])

    def create(self, request):
        return CreateRunView().post(request)


class SampleDataView(APIView):
    """Return a small canned payload to keep the existing sample route working."""

    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
