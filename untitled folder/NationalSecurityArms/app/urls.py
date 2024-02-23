from django.urls import path
from . import views
urlpatterns = [
path('',views.register,name="register"),
path('login/',views.user_login,name="user_login"),
path('dashboard/',views.dashboard,name="dashboard"),
path('logout/',views.logout,name="logout"),
path('admin_login/',views.admin_login,name="admin_login"),
path('admin_dashboard/',views.admin_dashboard,name="admin_dashboard"),
path('admin_logout/',views.admin_logout,name="admin_logout"),
path('add_job/', views.add_job,name="add_job"),
path('jobs/', views.jobs,name="jobs"),
path('edit_job/<int:pk>', views.edit_job,name="edit_job"),
path('delete_job/<int:pk>/', views.delete_job,name="delete_job"),
path('job_details/', views.job_details,name="job_details"),
path('apply/<int:pk>/', views.apply,name="apply"),
path('Applied/', views.Applied,name="Applied"),
path('applied_job_details/', views.applied_job_details,name="applied_job_details"),
path('admin_applied_job_details/', views.admin_applied_job_details,name="admin_applied_job_details"),
path('applicant/<int:pk>/', views.applicant,name="applicant"),
path('shortlist/<int:pk>/<int:job_id>/<int:user_id>/', views.shortlist,name="shortlist"),
path('send_status/', views.send_status,name="send_status"),
path('check_status/', views.check_status,name="check_status"),
path('user_check_status/', views.user_check_status,name="user_check_status"),
path('search/', views.search,name="search"),
path('search_resume/', views.search_resume,name="search_resume"),
]


