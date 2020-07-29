# Create your views here.
from django.shortcuts import render

from . import review_rate

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
    data = {'comment_str': comment_str, 'rate_str': rate_str, 'warn_str': warn_str, 'color': color}
    return render(request, "Front.html", context=data)
