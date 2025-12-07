from django.db import models


class Dataset(models.Model):
    name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name or f"Dataset {self.id}"
