# DLS_final_project (telegram: @gorrose)

### Основной проект - сегментация зданий (segmentation.ipynb)
1) **Проведена сегментация зданий**  
Детали и выводы предоставлены в ноутбуке segmentation.ipynb  
Для запуска ноутбуков необходимо скачать и разархивировать датасет [RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) в корневую папку  

2) **Проведено мини-исследование возможности предсказания высоты и соответственно площади зданий** исходя из нее в рамках ноутбука height_study.ipynb  
Для этого анализа высоты проведена отдельная ручная разметка некоторых фотографий в папках ./height_study/*

3) **Выполнена реализация приложения на Streamlit**:
На вход подается спутниковый снимок, а на выход - площадь застройки (строейний) в единицах измерения м^2  
Для запуска приложения:  
загрузить модель из [гугл-диска](https://drive.google.com/file/d/12QDGUwzNVX0AtFuqLxVqK-mu2JmYCqaP/view?usp=drive_link) и разместить ее в папке './checkpoints/'  
после в корневой папке выполнить команды:  
`pip install requirements.txt`  
`streamlit run app.py`  
Для тестирования лучше использовать фото из тестовой выборки, предоставленной в датасете RescueNet,  
либо самостоятельно подготовленную и размеченную тестовую выборку из папки **./manual_segmentation/Vologda**

5) Ручная разметка тестовой выборки предоставлена в папке **./manual_segmentation**

6) **Альтернативный проект - детекция зданий (detection.ipynb)**  - не используется для предсказания площади  
