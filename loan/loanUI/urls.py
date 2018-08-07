from django.conf.urls import url
from . import views

urlpatterns = [
    #/loan/
    url(r'^$', views.index, name='index'),
    #/loanUI/id>/
    url(r'^/detail/$',views.detail,name='detail'),
    url(r'^/detail/predict1$',views.predict1,name='predict1'),
    url(r'^/detail/predict2$',views.predict2,name='predict2'),
    ]


