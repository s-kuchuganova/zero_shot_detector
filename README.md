### ДЕТЕКЦИЯ ЦЕНТРА ОБЪЕКТА НА ФОТО

Для запуска выполните следующие команды внутри репозитория:

```bash
git clone --recurse-submodules https://github.com/s-kuchuganova/zero_shot_detector.git
cd zero_shot_detector
docker build -t detector .
docker run -v <your/path/to/tasks>:/tasks detector
```
Быстро ознакомиться с результатами и визуализацией можно в ноутбуке [Kuchuganova_zero-shot_object_detection_v2.ipynb](Kuchuganova_zero-shot_object_detection_v2.ipynb)


Что было добавлено:
- 
- уменьшен и зафиксирован размер входного изображения,
- добавлен датасет и даталоадер, обработка батчами,
- увеличена скорость обработки в 4.8 раз.

Предложения по дальнейшему улучшению:
- 
- конвертация модели в onnx/OpenVINO,
- препроцессинг изображений,
- отвязка от репозитория GroundingDINO.

