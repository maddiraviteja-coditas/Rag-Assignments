from django.urls import path
from .views import gpt, home, rag, rag_chat
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", home, name = "home"),
    path("rag", rag, name = "home"),
    path("rag/chat", rag_chat, name = "rag_chat"),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)