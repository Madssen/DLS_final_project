# DLS_final_project

### Основной проект - сегментация зданий (segmentation.ipynb)
1) **Проведена сегментация зданий**  
Детали и выводы предоставлены в ноутбуке  
Для запуска ноутбуков необходимо скачать и разархивировать датасет [RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) в корневую папку  

2) **Проведено мини-исследование возможности предсказания высоты и соответственно площади зданий** исходя из нее в рамках ноутбука height_study.ipynb  
Для этого анализа высоты проведена отдельная ручная разметка некоторых фотографий в папках ./height_study/*

3) **Выполнена реализация приложения на Streamlit**:
На вход подается спутниковый снимок, а на выход - площадь застройки (строейний) в единицах измерения м^2  
Для запуска приложения:  
`pip install requirements.txt`  
`streamlit run app.py`

**Альтернативный проект - детекция зданий (detection.ipynb)**  - не используется для предсказания площади  

