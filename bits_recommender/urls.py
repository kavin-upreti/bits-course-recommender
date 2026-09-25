from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path

from students import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("register/", views.register, name="register"),
    path("login/", auth_views.LoginView.as_view(template_name="students/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("profile/", views.profile, name="profile"),
    path("profile/electives/", views.electives, name="electives"),
    path("", views.home, name="home"),
]
