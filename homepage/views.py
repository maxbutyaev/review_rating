# Create your views here.
from django.shortcuts import render

from . import review_rate

server_adress='https://reviewrate.na4u.ru/'

def index(request):
    try:
        language = request.POST['language']
        text = request.POST['text']
    except:
        text = ''
    if text == '':
        comment_str, rate_str, warn_str, color= '', '', '', review_rate.color_dict['neutral']
    else:
        comment_str, rate_str, warn_str, color = review_rate.rate(text, language)
    data = {'comment_str': comment_str, 'rate_str': rate_str, 'warn_str': warn_str, 'color': color, 'server_adress': server_adress}
    return render(request, "Front_homepage.html", context=data)
