from django.db import models
import uuid
from django.core.exceptions import ValidationError


class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_example = models.BooleanField(default=False)
    example_type = models.CharField(max_length=50, blank=True, null=True)
    prompt = models.TextField(blank=True, null=True)

    def clean(self):
        if self.is_example:
            if not self.example_type:
                raise ValidationError(
                    {'example_type': 'Examples must have a type (e.g., Regression).'})
            if not self.prompt:
                raise ValidationError(
                    {'prompt': 'Examples must have a default prompt.'})
            if not self.name:
                raise ValidationError(
                    {'name': 'Examples must have a default name.'})

    def __str__(self):
        return self.name or f"Dataset {self.id}"
