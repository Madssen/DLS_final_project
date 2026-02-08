import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
import os
from calculate_area import predict_with_pixel_area

# Настройки
st.set_page_config(page_title="Анализ площади застройки", layout="wide")

# Заголовок
st.title("🏢 Анализ площади застройки")

# Автоматическая загрузка модели
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = torch.load('./checkpoints/best_model.pth', 
                          map_location=device,
                          weights_only=False)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

# Загружаем модель при старте
if 'model' not in st.session_state:
    with st.spinner("Загрузка модели..."):
        model, device = load_model()
        if model:
            st.session_state.model = model
            st.session_state.device = device

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
