from django.urls import include, path

from rest_framework.routers import DefaultRouter

from .views import CreateRunView, RunViewSet, SampleDataView

router = DefaultRouter()
router.register("runs", RunViewSet, basename="runs")

app_name = "api"

urlpatterns = [
    path("sample/", SampleDataView.as_view(), name="sample"),
    path("runs/create/", CreateRunView.as_view(), name="run-create"),
    path("", include(router.urls)),
]
