from rest_framework import serializers

from .models import Run
from .services import parse_csv_to_matrix

MAX_COLUMNS = 50000
MAX_ROWS = 500000


class RunInputSerializer(serializers.Serializer):
    dataset = serializers.CharField()
    prompt = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        dataset = attrs.get("dataset", "")
        try:
            headers, _, matrix = parse_csv_to_matrix(dataset)
        except ValueError as exc:
            raise serializers.ValidationError({"dataset": str(exc)})

        if len(headers) > MAX_COLUMNS or matrix.shape[0] > MAX_ROWS:
            raise serializers.ValidationError(
                {
                    "dataset": (
                        f"CSV too large. Limit is {MAX_ROWS:,} rows and "
                        f"{MAX_COLUMNS:,} columns."
                    )
                }
            )

        attrs["headers"] = headers
        attrs["dataset_matrix"] = matrix
        return attrs
