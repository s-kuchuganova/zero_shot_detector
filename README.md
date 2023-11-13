### ДЕТЕКЦИЯ ЦЕНТРА ОБЪЕКТА НА ФОТО

```bash
git clone --recurse-submodules https://github.com/s-kuchuganova/zero_shot_detector.git
cd zero_shot_detector
docker build -t detector .
docker run -v <your/path/to/tasks>:/tasks detector
```
Быстро ознакомиться с результатами и визуализацией можно в ноутбуке [Kuchuganova_zero-shot_object_detection.ipynb](Kuchuganova_zero-shot_object_detection.ipynb)

