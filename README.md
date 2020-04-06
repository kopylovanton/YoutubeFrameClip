# YoutubeFrameClip
Автоматизация сохранения движущихся объектов в кадре в виде изображений для бинарной классификации
Порядок работы:
- выбрать idt youtube
- установить максимальную площадь детектируемых объектов - по умолчанию 500 (для мячей)
- стартовать просмотр
- движущиеся объекты помечаются желтыми прямоугольниками. Для минимизации ложных срабатываний необходимо выбирать видео снятые неподвижной камерой
- остановить видео кнопкой пробел в интересующем месте ролика
- указать кликом левой кнопки мышки или тачпада на интересующий прямоугольник. Он будет подсвечен зеленым. Случайный соседний прямоугольник будет подсвечен красным
- нажать кнопку "s" для сохранени в памяти позитивного зеленого (1) и негативного красного (0) примера (горячие клавиши работают только в английской раскладке при отключенном Capslock)
- после сохранения в памяти продолжается воспроизведение ролика
- каждые 5 выборов происходит автоматическое сохранение в файл с именем "{}.pkl".format(ts.strftime("%Y-%m-%d_"+self.url)) 
- формат файла pickle, структура список элементов [1/0,RGB_array] 1 - для позитивных зеленых примеров, 0 для красных негативных

Программа использовалась для сбора изображений мяча в спортивных соревнованиях для задачи определения положения мяча в кадре при условии неподвижной камеры 

требуемые модули для python 3.7 перечислены в файле requirements.txt
newds.py основной исполняемый файл
Browseds.py простой браузер полученных файлов изображений
TL.ipynb transfer learning notebook (google colab) для полученных данных на основе TF Keras MobilenetV2 

-- примеры
Волейбол fdqOdTvGc9I (https://www.youtube.com/watch?v=fdqOdTvGc9I)
6G5-PMk_jjw
Минифутбол 

Баскет
QPuPxNyujXw