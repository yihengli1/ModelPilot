from rest_framework import serializers

from .models import Run


class RunSerializer(serializers.ModelSerializer):
    class Meta:
        model = Run
        fields = [
            "id",
            "dataset",
            "prompt",
            "context",
            "target_column",
            "results",
            "created_at",
        ]
        read_only_fields = ["id", "results", "created_at"]
