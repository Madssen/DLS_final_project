import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
import os
import requests
import re
from calculate_area import predict_with_pixel_area

# Настройки
st.set_page_config(page_title="Анализ площади застройки", layout="wide")

# Заголовок
st.title("🏢 Анализ площади застройки")

# ID файла на Google Drive
GOOGLE_DRIVE_FILE_ID = "12QDGUwzNVX0AtFuqLxVqK-mu2JmYCqaP"
MODEL_FILENAME = "best_model.pth"

def get_confirm_token(response):
    """Извлекает токен подтверждения из cookies"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Сохраняет содержимое ответа в файл с прогресс-баром"""
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Загружено: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Если файл уже существует, проверяем его размер
    if os.path.exists(MODEL_FILENAME):
        file_size = os.path.getsize(MODEL_FILENAME) / (1024 * 1024)
        if file_size > 1:  # Если файл больше 1MB, считаем его валидным
            try:
                model = torch.load(MODEL_FILENAME, map_location=device, weights_only=False)
                model.eval()
                st.success(f"✅ Модель загружена из кэша ({file_size:.1f} MB)")
                return model, device
            except:
                os.remove(MODEL_FILENAME)  # Удаляем поврежденный файл
    
    try:
        with st.spinner("🔄 Скачивание модели с Google Drive..."):
            session = requests.Session()
            
            # URL для скачивания
            URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
            
            # Первый запрос для получения токена подтверждения
            response = session.get(URL, stream=True)
            token = get_confirm_token(response)
            
            if token:
                # Если нужен токен подтверждения (большие файлы)
                params = {'id': GOOGLE_DRIVE_FILE_ID, 'confirm': token}
                response = session.get(URL, params=params, stream=True)
            
            # Сохраняем файл
            save_response_content(response, MODEL_FILENAME)
            
            # Проверяем размер скачанного файла
            if os.path.exists(MODEL_FILENAME):
                file_size = os.path.getsize(MODEL_FILENAME) / (1024 * 1024)
                if file_size < 1:
                    st.error("❌ Скачанный файл слишком мал. Возможно, это HTML-страница.")
                    os.remove(MODEL_FILENAME)
                    return None, device
                
                st.success(f"✅ Модель скачана! Размер: {file_size:.1f} MB")
                
                # Загружаем модель
                model = torch.load(MODEL_FILENAME, map_location=device, weights_only=False)
                model.eval()
                return model, device
            else:
                st.error("❌ Не удалось скачать файл")
                return None, device
                
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {str(e)}")
        # Показываем альтернативные инструкции
        st.info("""
        **Если скачивание не работает:**
        
        1. **Откройте эту ссылку в браузере:**  
           https://drive.google.com/uc?export=download&id=12QDGUwzNVX0AtFuqLxVqK-mu2JmYCqaP
        
        2. **Вручную скачайте файл `best_model.pth`**
        
        3. **Загрузите его прямо в Streamlit Cloud:**
           ```python
           # Временно: загрузите файл через интерфейс Streamlit
           uploaded_model = st.file_uploader("Загрузите модель", type=['pth'])
           if uploaded_model:
               with open('best_model.pth', 'wb') as f:
                   f.write(uploaded_model.getbuffer())
           ```
        """)
        return None, device

# Загружаем модель при старте
if 'model' not in st.session_state:
    model, device = load_model()
    if model:
        st.session_state.model = model
        st.session_state.device = device
    else:
        # Показываем кнопку для повторной попытки
        if st.button("🔄 Повторить загрузку модели"):
            st.rerun()

# ========== БОКОВАЯ ПАНЕЛЬ ==========
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    
    # Раздел: Загрузка модели
    st.markdown("### 🌀 Загрузка модели")
    if 'model' in st.session_state:
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.success(f"Модель загружена на: **{device_name}**")
    else:
        st.error("❌ Модель не загружена")
    
    st.markdown("---")
    
    # Раздел: Параметры площади
    st.markdown("### 📐 Параметры площади")
    pixel_area = st.number_input(
        "Площадь пикселя (м²/пикс)",
        value=0.000451, 
        min_value=0.000001,
        max_value=10.0,
        step=0.000001,
        format="%.6f",
        help="Пример: если 1 пиксель = 0.0212×0.0212 метра, то площадь = 0.000451 м²"
    )
    
    side_length = pixel_area ** 0.5
    st.info(f"**Текущее значение:** {pixel_area:.6f} м²/пикс")
    st.info(f"**Сторона пикселя:** {side_length*100:.2f} см")

# ========== ГЛАВНАЯ ОБЛАСТЬ ==========
st.header("📤 Загрузка изображения")

uploaded_file = st.file_uploader("Выберите спутниковый снимок", 
                                 type=['jpg', 'png', 'jpeg', 'tif', 'tiff'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    if 'model' in st.session_state:
        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(tmp.name, format='JPEG', quality=95)
        
        with st.spinner("Выполняется анализ..."):
            results = predict_with_pixel_area(
                st.session_state.model,
                tmp.name,
                st.session_state.device,
                pixel_area_m2=pixel_area
            )
        
        if results:
            # Две колонки
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Исходное изображение")
                original_width = results['original_width']
                original_height = results['original_height']
                st.image(image, caption=f"{uploaded_file.name} ({original_width}×{original_height})", width=500)
            
            with col2:
                st.subheader("🎯 Сегментация зданий")
                
                # Создаем цветную маску (используем pred_mask_original)
                color_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)
                color_mask[results['pred_mask_original'] == 1] = [255, 0, 0]
                
                # Наложение с прозрачностью
                overlay = results['image_rgb'].copy()
                alpha = 0.6
                overlay = cv2.addWeighted(overlay, 1-alpha, color_mask, alpha, 0)
                
                st.image(overlay, 
                        caption=f"Обнаружено зданий: {results['num_buildings']}",
                        width=500)
            
            # Результаты
            st.markdown("---")
            st.subheader("📊 Результаты анализа")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Площадь застройки", f"{results['total_area_m2']:.1f} м²")
            
            with col_b:
                st.metric("Количество зданий", results['num_buildings'])
            
            with col_c:
                st.metric("Пикселей зданий (256×256)", f"{results['pixels_count_small']:,}")
            
            # Детали расчета
            with st.expander("🔍 Подробности расчета", expanded=False):
                st.write(f"**Размер изображения:** {results['original_width']}×{results['original_height']} пикс.")
                st.write(f"**Площадь пикселя:** {results['pixel_area_m2']:.6f} м²")
                st.write(f"**Коэффициент масштабирования:** {results['area_scale_factor']:.2f}")
                st.write(f"**Масштабированных пикселей:** {results['pixels_count_original']:,.0f}")
                st.write(f"**Расчет:** {results['pixels_count_original']:,.0f} пикс × {results['pixel_area_m2']:.6f} м²/пикс")
        
        # Удаляем временный файл
        try:
            os.unlink(tmp.name)
        except:
            pass
    else:
        st.warning("Модель не загружена. Невозможно выполнить анализ.")
        st.image(image, caption=f"Загружено: {uploaded_file.name}", width=500)
else:
    st.info("👆 Загрузите спутниковый снимок для анализа")
