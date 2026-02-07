# DLS_final_project

### Основной проект
1) **Проведена сегментация зданий (segmentation.ipynb)**  
Детали и выводы предоставлены в ноутбуке
Для запуска ноутбуков необходимо скачать и разархивировать датасет [RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation)  в корневую папку 

2) **Проведено мини-исследование возможности предсказания высоты и соответственно площади зданий** исходя из нее в рамках ноутбука height_study.ipynb
Для этого анализа высоты проведена отдельная ручная разметка некоторых фотографий в папках ./height_study/*

3) **Выполнена реализация приложения на Streamlit**
Для запуска приложения:  
>>> `pip install requirements.txt`
>>> `streamlit run app.py`

### Альтернативный проект - детекция (detection.ipynb) 
Не используется для предсказания площади
Детали и выводы предоставлены в ноутбуке
