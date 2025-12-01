from typing import List, Tuple

import numpy as np
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from .services import _coerce_value, parse_csv_to_matrix


class CreateRunView(APIView):
    def post(self, request):
        dataset = request.data.get("dataset", "")
        prompt = request.data.get("prompt", "")
        context = request.data.get("context", "")

        print("dataset", dataset)
        print("prompt", prompt)
        print("context", context)

        try:
            headers, rows, matrix = parse_csv_to_matrix(dataset)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            return Response(
                {"error": "Failed to parse CSV content."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response([], status=status.HTTP_200_OK)

        response_payload = {
            "prompt": prompt,
            "context": context,
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
    def list(self, request):
        return Response([])

    def create(self, request):
        return CreateRunView().post(request)


class SampleDataView(APIView):

    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
