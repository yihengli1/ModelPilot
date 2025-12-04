from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .contexts import TESTING_CONTEXT
from .serializers import RunInputSerializer
from .services import generate_plan_from_gpt
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
        prompt = serializer.validated_data.get("prompt", "")
        llm_result = training_pipeline(TESTING_CONTEXT, prompt, dataset_matrix)

        response_payload = {
            "prompt": prompt,
            "context": TESTING_CONTEXT,
            "dataset": request.data.get("dataset", ""),
            "llm_result": llm_result
        }

        return Response(response_payload, status=status.HTTP_200_OK)


class SampleDataView(APIView):
    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
