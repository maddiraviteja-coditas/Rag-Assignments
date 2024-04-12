from django.urls import path
from .views import gpt, home, rag, rag_chat


urlpatterns = [
    path("", home, name = "home"),
    path("rag", rag, name = "home"),
    path("rag/chat", rag_chat, name = "rag_chat"),
]