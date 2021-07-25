# Модель для классификации отзывов
### В этом репозитории содержатся все файлы Django-проекта веб-сервиса для классификации отзывов и присвоения им рейтинга от 1 до 10
### Работу прототипа веб-сервиса можно оценить [ЗДЕСЬ*](https://reviewrating.na4u.ru/). Кроме того, можно протестировать модель самому и получить все приведённые ниже метрики [ЗДЕСЬ](https://colab.research.google.com/drive/1-hNOFUq7I6_1qT7jcZEKT05_W0X3j37H#scrollTo=H9jD3AKKqzWe).
##### *Прототип веб-сервиса более не поддерживается, т.к. хостинг стоит денег :) 
# Отчёт о проделанной работе, оценка точности модели.
## I.  Выполнение
### 1)  [Предобработка данных](https://github.com/maxbutyaev/review_rating/blob/master/data/gettingDatasets.py)
- Для начала данные из 50,000 .txt документов были предобработаны в один датасет, удобный для последующей обработки
- При помощи библиотек `bs4` и `nltk` предложения были разбиты на слова, убраны html-тэги, местоимения и прочие незначащие слова. Оставшиеся слова лемматизированы и токенизированы (однокоренные слова сводятся таким образом к одинаковому значению без окончаний, что упрощает дальнейшую обработку, хотя некоторые токены уже не представляют из себя грамматически верное слово)
#### В итоге получаем датасет с id каждого комментария, его оценкой по 10-балльной шкале и комментарием, состоящим только из корней важных слов этого комментария.
### 2)  [Создание модели](https://github.com/maxbutyaev/review_rating/blob/master/data/creating_model.py). Эти пункты применялись только к тренировочной выборке из 25,000 комментариев.
- При помощи CountVectorizer все комментарии векторизуются: Слова, встречающиеся в комментариях редко удаляются. Оставляем словарь из 5000 слов и каждый комментарий представляем в виде вектора в 5000-мерном пространстве, где n-ой координатой является количество n-го слова в этом комментарии.
- На получившемся датасете обучаем модель `XGBRegressor` и находим его оптимальные параметры(я варьировал максимальную глубину отдельных деревьев и их количество) при помощи кросс-валидации(`RandomizedSearchCV` и `GridSearchCV`)

    
## II.  Оценка точности модели
### 1)  Точность классификации по баллам от 1 до 10:
- Моделями, показавшими себя лучше всех на кросс-валидации предсказываем результат на тестовой выборке.
- Так как наша модель - регрессор, она предсказывает не целые числа и её предсказания требуют округления.
- Для оценки точности модели можно использовать различные метрики. Приведу результаты некоторых из них для своей модели:
    - Среднее абсолютное отклонение: `mean_absolute_error = 1.79288`
    - Коэффициент детерминации: `r2_score = 0.5719187875670133`
    - Максимальное абсолютное отклонение: `max_error = 9.0`
#### Посмотрев на последнее значение можно испугаться за точность нашей модели. Чтобы она не была настолько страшной попробуем визуализировать результаты в виде гистограммы, показывающей зависимость количества отзывов от абсолютного отклонения предсказанного значения от фактического. 
___
![](https://sun1-98.userapi.com/M-GOshZCLUdhqEevcNORqZHcjsUzfhE5PjIIIA/LsHdzEdnwIM.jpg)
#### Получаем, что 17.5% результатов абсолютно верные, почти половина - отличаются не более чем на один балл, 73% - не более чем на два. Из 25,000 отзывов тестовой выборки лишь 1.5% отличаются более чем на 5 баллов. Из них 3 отзыва отличаются на 9. Если открыть эти отзывы, то окажется, что данные фильмы настолько ужасны, что даже понравились авторам рецензий, но в тексте большая часть уделяется всё же их ужасности:) Считаю подобные выбросы вполне логичными и нормальными для таких данных, поэтому `max_error = 9.0` не показатель неточности нашей модели. Также встречаются случаи ошибки в выставлении оценки(файл комментария оценен в 10 баллов, а в тексте автор пишет, что оценивает фильм заметно ниже)
### 2)  Точность классификации на положительные/отрицательные:
- Проставим положительные маркеры для отзывов с оценкой >5 баллов и отрицательные для отзывов с оценкой <=5
- Проверим, как справился наш регрессор с бинарной классификацией. Для этого используем стандартные метрики:
    - `accuracy = 0.853`
    - `precision = 0.8415512036535335`
    - `recall = 0.86976`
    - `f1 = 0.8554231086982179`
- Визуализируем результат с помощью `confusion_matrix`:
___
![](https://sun1-27.userapi.com/cmIJesKlIocFf91BxBivTV2lZtX22bNuxT9XpQ/wM2GzS_J_NA.jpg)
## III.  Вывод
- По всем метрикам модель не идеальна, но показывает довольно неплохие результаты.
- Для упрощения модели можно попробовать использовать векторизацию и кластеризацию слов по характеру. Таким образом можно будет заметно уменьшить число фич для модели (сейчас их 5000) и ускорить процесс создания модели (моя модель обучалась около 4-5 часов)
