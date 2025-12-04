from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .contexts import TESTING_CONTEXT
from .serializers import RunInputSerializer
from .services import generate_plan_from_gpt


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

        validated_data = serializer.validated_data
        dataset_matrix = validated_data["dataset_matrix"]
        prompt = validated_data.get("prompt", "")

        if dataset_matrix.size == 0:
            return Response({"error": "Dataset CSV cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            llm_result = generate_plan_from_gpt(
                system_context=TESTING_CONTEXT,
                prompt=prompt,
                dataset=dataset_matrix,
            )
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except EnvironmentError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except ImportError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception:
            return Response(
                {"error": "Failed to generate plan from LLM."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        response_payload = {
            "prompt": prompt,
            "context": TESTING_CONTEXT,
            "dataset": validated_data.get("dataset", ""),
            "llm_result": llm_result
        }

        return Response(response_payload, status=status.HTTP_200_OK)


class SampleDataView(APIView):
    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
