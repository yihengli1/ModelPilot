from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Run
from .serializers import RunSerializer
from .services import analyze_dataset, load_sample_dataset, load_sample_prompt


class RunViewSet(viewsets.ModelViewSet):
    queryset = Run.objects.all()
    serializer_class = RunSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        results = analyze_dataset(
            dataset=serializer.validated_data.get("dataset", ""),
            prompt=serializer.validated_data.get("prompt", ""),
            context=serializer.validated_data.get("context", ""),
            target_column=serializer.validated_data.get("target_column", ""),
        )
        self.perform_create(serializer, results)
        headers = self.get_success_headers(serializer.data)
        payload = serializer.data
        payload["results"] = results
        return Response(payload, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer, results):
        serializer.save(results=results)

    @action(methods=["post"], detail=False, url_path="create")
    def create_run(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)


class SampleDataView(APIView):

    def get(self, request):
        return Response(
            {"dataset": load_sample_dataset(), "prompt": load_sample_prompt()}
        )
