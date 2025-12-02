from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from .contexts import SYSTEM_CONTEXT, TESTING_CONTEXT
from .services import _coerce_value, parse_csv_to_matrix, generate_plan_from_gpt


class CreateRunView(APIView):
    def post(self, request):
        dataset = request.data.get("dataset", "")
        prompt = request.data.get("prompt", "")

        if not dataset.strip():
            return Response({"error": "Dataset CSV cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            llm_result = generate_plan_from_gpt(
                system_context=TESTING_CONTEXT,
                prompt=prompt,
                dataset=dataset,
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
            "dataset": dataset,
            "llm_result": llm_result
        }

        return Response(response_payload, status=status.HTTP_200_OK)


class SampleDataView(APIView):

    def get(self, request):
        return Response({"message": "Sample endpoint", "data": []})
