from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction

from api.models import Dataset

DATASETS = [
    {
        "name": "US Elections",
        "type": "Classification",
        "description": "Predict Red vs Blue states based on demographics.",
        "filename": "smallCities.csv",
        "prompt": "Target Column = y. Republican vs Democratic States in U.S. Use a decision tree to classify states."
    },
    {
        "name": "Logistic Data",
        "type": "Classification",
        "description": "High-dimensional binary classification dataset with 100 features.",
        "filename": "logisticData.csv",
        "prompt": "Target Column = y. Binary classification problem. Predict the target y (-1 or 1) based on the 100 numerical features."
    },
    {
        "name": "Multi-Class Data",
        "type": "Classification",
        "description": "Multi-class classification dataset with 2 features and multiple classes.",
        "filename": "multiData.csv",
        "prompt": "Target Column = y. Multi-class classification. Predict the class (0, 1, 2, 3...) based on the two input features."
    },
    {
        "name": "Basis Regression",
        "type": "Regression",
        "description": "Simple 1D regression dataset.",
        "filename": "basisData.csv",
        "prompt": "Target Column = y. Regression problem. Predict the value of y based on feature 0."
    },
    {
        "name": "Student Performance",
        "type": "Regression",
        "description": "Predict student final grades based on social, gender, and study data.",
        "filename": "student-mat.csv",
        "prompt": "Target Column = G3. Predict the student's final grade (G3) based on attributes like study time, failures, and previous grades (G1, G2)."
    },
    {
        "name": "Customer Segmentation (Clustering)",
        "type": "Clustering",
        "description": "Group customers into natural segments based on spending behavior and engagement features.",
        "filename": "clusterData.csv",
        "prompt": "No target column. Perform clustering to discover meaningful clusters."
    }
]


class Command(BaseCommand):
    help = "Seed (upsert) example datasets."

    @transaction.atomic
    def handle(self, *args, **kwargs):
        base = Path(settings.BASE_DIR) / "test_datasets"
        created, updated, skipped = 0, 0, 0

        for ex in DATASETS:
            csv_path = base / ex["filename"]
            if not csv_path.exists():
                self.stdout.write(self.style.WARNING(
                    f"[SKIP] Missing file: {csv_path}"))
                skipped += 1
                continue

            obj, was_created = Dataset.objects.update_or_create(
                name=ex["name"],
                defaults={
                    "description": ex["description"],
                    "prompt": ex["prompt"],
                    "is_example": True,
                    "example_type": ex["type"],
                },
            )

            with open(csv_path, "rb") as f:
                obj.file.save(ex["filename"], File(f), save=True)

            created += int(was_created)
            updated += int(not was_created)

            self.stdout.write(self.style.SUCCESS(
                f"[OK] {'Created' if was_created else 'Updated'}: {obj.name}"
            ))

        self.stdout.write(self.style.SUCCESS(
            f"Done. created={created} updated={updated} skipped={skipped}"
        ))
