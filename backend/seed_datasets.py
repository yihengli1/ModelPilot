import requests
import os

BASE_URL = "http://127.0.0.1:8000/api/datasets/"
SOURCE_DIR = "../test_datasets"

# UNCOMMENT BELOW TO SEED DATASETS

datasets_to_upload = [
    # {
    #     "name": "US Elections",
    #     "type": "Classification",
    #     "description": "Predict Red vs Blue states based on demographics.",
    #     "filename": "smallCities.csv",
    #     "prompt": "Target Column = y. Republican vs Democratic States in U.S. Use a decision tree to classify states."
    # },
    # {
    #     "name": "Logistic Data",
    #     "type": "Classification",
    #     "description": "High-dimensional binary classification dataset with 100 features.",
    #     "filename": "logisticData.csv",
    #     "prompt": "Target Column = y. Binary classification problem. Predict the target y (-1 or 1) based on the 100 numerical features."
    # },
    # {
    #     "name": "Multi-Class Data",
    #     "type": "Classification",
    #     "description": "Multi-class classification dataset with 2 features and multiple classes.",
    #     "filename": "multiData.csv",
    #     "prompt": "Target Column = y. Multi-class classification. Predict the class (0, 1, 2, 3...) based on the two input features."
    # },
    # {
    #     "name": "Basis Regression",
    #     "type": "Regression",
    #     "description": "Simple 1D regression dataset.",
    #     "filename": "basisData.csv",
    #     "prompt": "Target Column = y. Regression problem. Predict the value of y based on feature 0."
    # },
    # {
    #     "name": "Student Performance",
    #     "type": "Regression",
    #     "description": "Predict student final grades based on social, gender, and study data.",
    #     "filename": "student-mat.csv",
    #     "prompt": "Target Column = G3. Predict the student's final grade (G3) based on attributes like study time, failures, and previous grades (G1, G2)."
    # },
    # {
    #     "name": "Customer Segmentation (Clustering)",
    #     "type": "Clustering",
    #     "description": "Group customers into natural segments based on spending behavior and engagement features.",
    #     "filename": "clusterData.csv",
    #     "prompt": "No target column. Perform clustering to discover meaningful clusters."
    # }


]


def upload_datasets():
    print(f"Starting upload to {BASE_URL}...")

    for item in datasets_to_upload:
        file_path = os.path.join(SOURCE_DIR, item["filename"])

        if not os.path.exists(file_path):
            print(f"[SKIPPING] File not found: {file_path}")
            continue

        payload = {
            "name": item["name"],
            "description": item["description"],
            "prompt": item["prompt"],
            "is_example": "true",
            "example_type": item["type"]
        }

        # Open file in binary mode
        try:
            with open(file_path, 'rb') as f:
                files = {
                    'file': (item["filename"], f, 'text/csv')
                }

                response = requests.post(BASE_URL, data=payload, files=files)

                if response.status_code == 201:
                    print(f"[SUCCESS] Uploaded: {item['name']}")
                else:
                    print(
                        f"[FAILED] {item['name']} - Status: {response.status_code}")
                    print(f"Response: {response.text}")

        except Exception as e:
            print(f"[ERROR] Could not process {item['name']}: {e}")


if __name__ == "__main__":
    upload_datasets()
