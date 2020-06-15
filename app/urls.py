from django.urls import path

from . import views

urlpatterns = [
     path('', views.index, name='index'),
     path('404', views.handler404, name='404'),
     path('500', views.handler500, name='500'),
     path('register', views.RegisterFormView.as_view(), name='register'),
     path('login', views.LoginFormView.as_view(), name='login'),
     path('logout', views.logoutForm, name='logout'),
     path('register/', views.RegisterFormView.as_view(), name='registerslash'),
     path('login/', views.LoginFormView.as_view(), name='loginslash'),
     path('logout/', views.logoutForm, name='logoutslash'),
     path('statcovid', views.CovidStatScript, name="CovidStatScript"),
     path('news-10',views.retrieve_top10_news, name="retrieve_top10_news")
]