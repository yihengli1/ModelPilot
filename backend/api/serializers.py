from rest_framework import serializers

from .models import Run
from .services import parse_csv_to_matrix


class RunInputSerializer(serializers.Serializer):
    dataset = serializers.CharField()
    prompt = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        dataset = attrs.get("dataset", "")
        try:
            _, _, matrix = parse_csv_to_matrix(dataset)
        except ValueError as exc:
            raise serializers.ValidationError({"dataset": str(exc)})
        attrs["dataset_matrix"] = matrix
        return attrs
