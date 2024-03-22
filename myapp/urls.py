from django.urls import path
from .views import register_view, login_view, logout_view  # 确保导入了新的视图

urlpatterns = [
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    # 其他URLs
]
