from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .contexts import TESTING_CONTEXT
from .serializers import RunInputSerializer
from .modelling import training_pipeline


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


class SampleDataView(APIView):
    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
