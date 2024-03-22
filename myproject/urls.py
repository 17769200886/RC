from myapp import views
from django.contrib import admin  # 确保有这一行来导入admin
from django.urls import path, include  # 导入path和include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('myapp/', include('myapp.urls')),
    path('', include('myapp.urls')),
    # 其他 URL 配置
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('probability_result/', views.probability_result, name='probability_result'),
    path('shap_values_result/', views.shap_values_result, name='shap_values_result'),
    path('shap_force_plots_result/', views.shap_force_plots_result, name='shap_force_plots_result'),
    path('shap_values_data/', views.shap_values_data, name='shap_values_data'),
]


