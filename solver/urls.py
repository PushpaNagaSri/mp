from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.graphical_view, name='graphical_view'),
    path('simplex/', views.simplex_view, name='simplex_view'),
    path('transportation/', views.transportation_view, name='transportation_view'),
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
