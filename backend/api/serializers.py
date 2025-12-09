from rest_framework import serializers

from .services import parse_csv_to_matrix
from .models import Dataset

MAX_COLUMNS = 1000
MAX_ROWS = 500000
MAX_WORD_COUNT = 500


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'file', 'uploaded_at',
                  'is_example', 'example_type', 'prompt', 'description']
        read_only_fields = ['id', 'uploaded_at']

    def validate_file(self, value):
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("Only CSV files are allowed.")
        return value

    def validate(self, data):
        is_example = data.get(
            'is_example', self.instance.is_example if self.instance else False)

        if is_example:
            prompt = data.get(
                'prompt', self.instance.prompt if self.instance else None)
            example_type = data.get(
                'example_type', self.instance.example_type if self.instance else None)
            name = data.get(
                'name', self.instance.name if self.instance else None)
            description = data.get(
                'name', self.instance.description if self.instance else None)

            if not prompt:
                raise serializers.ValidationError(
                    {"prompt": "This field is required when is_example is True."})
            if not example_type:
                raise serializers.ValidationError(
                    {"example_type": "This field is required when is_example is True."})
            if not name:
                raise serializers.ValidationError(
                    {"name": "This field is required when is_example is True."})
            if not description:
                raise serializers.ValidationError(
                    {"description": "This field is required when is_example is True."})

        return data


class RunInputSerializer(serializers.Serializer):
    dataset = serializers.CharField()
    prompt = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        dataset = attrs.get("dataset", "")
        try:
            headers, _, matrix = parse_csv_to_matrix(dataset)
        except ValueError as exc:
            raise serializers.ValidationError({"dataset": str(exc)})

        word_count = len(self.prompt.strip().split())

        if word_count > MAX_WORD_COUNT:
            raise serializers.ValidationError(
                f"Prompt exceeds the {MAX_WORD_COUNT} word limit. (Current: {word_count} words)"
            )

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
