from django.urls import include, path

from rest_framework.routers import DefaultRouter

from .views import CreateRunView, DatasetListCreateView, DatasetDetailView, ExampleDatasetListView, HealthCheckView

router = DefaultRouter()

app_name = "api"

urlpatterns = [
    path("run-prompt/", CreateRunView.as_view(), name="run-prompt"),
    path("datasets/", DatasetListCreateView.as_view(), name="dataset-list-create"),
    path("datasets/<uuid:pk>/",
         DatasetDetailView.as_view(), name="get-dataset"),
    path("datasets/examples/", ExampleDatasetListView.as_view(),
         name="get-example-datasets"),
    path("", include(router.urls)),
    path("health/", HealthCheckView.as_view(), name="health-check"),
]
