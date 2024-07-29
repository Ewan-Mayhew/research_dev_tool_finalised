from django.urls import path, include
from . import views
from .views import trends_closest_view, trends_scatter_view
from django.contrib import admin
from django.contrib.auth import views as auth_views
from .views import run_populators
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.paper_list, name='paper_list'),
    path('paper/<int:pk>/', views.paper_detail, name='paper_detail'),
    path('upload_json/', views.upload_json, name='upload_json'),
    path('run_arxiv_script/', views.run_arxiv_script, name='run_arxiv_script'),
    path('upload_papers/', views.upload_papers, name='upload_papers'),
    path('search_papers/', views.search_papers, name='search_papers'),
    path('fetch-papers/', views.fetch_and_process_papers, name='fetch_papers'),
    path('trends/closest/', views.trends_closest_view, name='trends-closest'),
    path('trends/scatter/', views.trends_scatter_view, name='trends-scatter'),
    path('fetch_and_trends/', views.fetch_and_trends_view, name='fetch_and_trends'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('run-populators/', run_populators, name='run_populators'),
    path('paper_list/', views.paper_list, name='paper_list'),
    path('run_arxiv_query/', views.run_arxiv_query, name='run_arxiv_query'),
    path('about/', views.about, name='about'),
]
