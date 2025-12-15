import sys
import csv
import io
from rest_framework import serializers

from .services import parse_csv_to_matrix
from .models import Dataset

MAX_COLUMNS = 1000
MAX_ROWS = 500000
MAX_WORD_COUNT = 500
MAX_FILE_SIZE_MB = 10


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'file', 'uploaded_at',
                  'is_example', 'example_type', 'prompt', 'description']
        read_only_fields = ['id', 'uploaded_at']
        extra_kwargs = {
            'file': {'required': False}
        }

    def validate_file(self, value):
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("Only CSV files are allowed.")
        return value

    def validate(self, data):
        if self.instance is None and not data.get('file'):
            raise serializers.ValidationError(
                {"file": "File is required for creation."})

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

        if len(dataset.encode('utf-8')) > (MAX_FILE_SIZE_MB * 1024 * 1024):
            raise serializers.ValidationError(
                {"dataset": f"File too large. Max size is {MAX_FILE_SIZE_MB}MB."}
            )

        try:
            f = io.StringIO(dataset)
            reader = csv.reader(f)
            try:
                header = next(reader)
                if len(header) > MAX_COLUMNS:
                    raise serializers.ValidationError(
                        {"dataset": f"Too many columns. Max is {MAX_COLUMNS}."}
                    )
            except StopIteration:
                raise serializers.ValidationError({"dataset": "Empty file."})

            # Check rows (iterating is cheaper than loading all at once)
            row_count = 0
            for _ in reader:
                row_count += 1
                if row_count > MAX_ROWS:
                    raise serializers.ValidationError(
                        {"dataset": f"Too many rows. Max is {MAX_ROWS}."}
                    )

        except csv.Error:
            raise serializers.ValidationError(
                {"dataset": "Invalid CSV format."})

        try:
            headers, _, matrix = parse_csv_to_matrix(dataset)
        except ValueError as exc:
            raise serializers.ValidationError({"dataset": str(exc)})

        word_count = len(attrs.get("prompt").strip().split())

        if word_count > MAX_WORD_COUNT:
            raise serializers.ValidationError(
                f"Prompt exceeds the {MAX_WORD_COUNT} word limit. (Current: {word_count} words)"
            )

        attrs["headers"] = headers
        attrs["dataset_matrix"] = matrix
        return attrs
