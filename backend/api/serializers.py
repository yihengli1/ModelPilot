import csv
import io
from rest_framework import serializers

from .services import parse_csv_to_matrix
from .models import Dataset

MAX_COLUMNS = 1000
MAX_ROWS = 500000
MAX_WORD_COUNT = 500
MAX_FILE_SIZE_MB = 10


class RunInputSerializer(serializers.Serializer):
    dataset_file = serializers.FileField()
    prompt = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        f = attrs["dataset_file"]

        if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise serializers.ValidationError(
                {"dataset_file": f"File too large. Max is {MAX_FILE_SIZE_MB}MB."}
            )

        raw = f.read()
        text = raw.decode("utf-8")

        try:
            reader = csv.reader(io.StringIO(text))
            header = next(reader)
            if not header:
                raise serializers.ValidationError(
                    {"dataset_file": "Missing header row."})
        except StopIteration:
            raise serializers.ValidationError({"dataset_file": "Empty file."})
        except csv.Error:
            raise serializers.ValidationError(
                {"dataset_file": "Invalid CSV format."})

        prompt = (attrs.get("prompt") or "").strip()
        if len(prompt.split()) > MAX_WORD_COUNT:
            raise serializers.ValidationError(
                {"prompt": f"Prompt exceeds {MAX_WORD_COUNT} words."}
            )

        headers, matrix = parse_csv_to_matrix(text)

        attrs["headers"] = headers
        attrs["dataset_matrix"] = matrix
        attrs["prompt"] = prompt
        return attrs


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
                'description', self.instance.description if self.instance else None)

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
