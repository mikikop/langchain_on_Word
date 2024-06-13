from django.urls import path
from .views import QuestionAnswerView

urlpatterns = [
    path('ask/', QuestionAnswerView.as_view(), name='ask-a-question'),
]