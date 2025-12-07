from rest_framework import serializers

from .services import parse_csv_to_matrix
from .models import Dataset

MAX_COLUMNS = 1000
MAX_ROWS = 500000


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'file', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']

    def validate_file(self, value):
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("Only CSV files are allowed.")
        return value


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
