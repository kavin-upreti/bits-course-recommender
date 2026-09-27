from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path

from recommender import views as recommender_views
from students import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("register/", views.register, name="register"),
    path("login/", auth_views.LoginView.as_view(template_name="students/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("profile/", views.profile, name="profile"),
    path("profile/electives/", views.electives, name="electives"),
    path("profile/electives/remaining/", views.remaining_preview, name="remaining_preview"),
    path("semester/", views.semester, name="semester"),
    path("course/<str:code>/", views.course, name="course"),
    path("chat/", recommender_views.chat, name="chat"),
    path("recommend/", recommender_views.recommend, name="recommend"),
    path("plan/select/", recommender_views.select_course, name="plan_select"),
    path("plan/timetables/", recommender_views.plan_timetables, name="plan_timetables"),
    path("plan/finalise/", recommender_views.finalise, name="plan_finalise"),
    path("semester/remove/", views.remove_course, name="remove_course"),
    path("", views.home, name="home"),
]
