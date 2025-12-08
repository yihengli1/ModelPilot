from django.urls import include, path

from rest_framework.routers import DefaultRouter

from .views import CreateRunView, DatasetListCreateView, DatasetDetailView

router = DefaultRouter()

app_name = "api"

urlpatterns = [
    path("run-prompt/", CreateRunView.as_view(), name="run-prompt"),
    path("datasets/", DatasetListCreateView.as_view(), name="dataset-list-create"),
    path("datasets/<int:pk>/",
         DatasetDetailView.as_view(), name="upload-dataset"),
    path("", include(router.urls)),
]
