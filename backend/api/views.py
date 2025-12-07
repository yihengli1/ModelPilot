from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import RunInputSerializer, DatasetSerializer
from .pipeline import training_pipeline
from .models import Dataset


class CreateRunView(APIView):
    def post(self, request):

        serializer = RunInputSerializer(
            data={
                "dataset": request.data.get("dataset", ""),
                "prompt": request.data.get("prompt", ""),
            }
        )

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        dataset_matrix = serializer.validated_data["dataset_matrix"]
        headers = serializer.validated_data.get("headers")
        prompt = serializer.validated_data.get("prompt", "")
        final_results = training_pipeline(
            prompt, dataset_matrix, headers=headers)

        response_payload = {
            "prompt": prompt,
            "dataset": request.data.get("dataset", ""),
            "plan": final_results["plan"],
            "final_results": final_results["results"],
        }

        return Response(response_payload, status=status.HTTP_200_OK)


class UploadDatasetView(generics.CreateAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    # Important for handling files
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            dataset_instance = serializer.save()

            return Response({
                "message": "File uploaded successfully",
                "id": dataset_instance.id,
                "url": dataset_instance.file.url,
                # "preview": preview
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SampleDataView(APIView):
    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
