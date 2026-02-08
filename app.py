import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
import os
import requests
from calculate_area import predict_with_pixel_area

# Настройки
st.set_page_config(page_title="Анализ площади застройки", layout="wide")

# Заголовок
st.title("🏢 Анализ площади застройки")

# Прямая ссылка на скачивание модели
MODEL_URL = "https://drive.google.com/uc?export=download&id=12QDGUwzNVX0AtFuqLxVqK-mu2JmYCqaP"
MODEL_FILENAME = "best_model.pth"

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Проверяем, есть ли уже скачанная модель
        if not os.path.exists(MODEL_FILENAME):
            with st.spinner("📥 Скачивание модели..."):
                # Скачиваем файл напрямую
                response = requests.get(MODEL_URL, stream=True, timeout=30)
                
                # Проверяем статус
                if response.status_code != 200:
                    st.error(f"Ошибка HTTP: {response.status_code}")
                    return None, device
                
                # Проверяем, что это не HTML страница
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type.lower():
                    # Читаем немного контента чтобы понять что это
                    content_preview = response.content[:200]
                    st.error(f"Получена HTML страница вместо файла. Первые 200 байт: {content_preview}")
                    
                    # Пробуем альтернативный формат
                    st.info("Пробую альтернативный URL...")
                    alt_url = f"https://docs.google.com/uc?export=download&id=12QDGUwzNVX0AtFuqLxVqK-mu2JmYCqaP&confirm=t"
                    response = requests.get(alt_url, stream=True, timeout=30)
                
                # Получаем размер файла
                total_size = int(response.headers.get('content-length', 0))
                
                if total_size < 1024 * 1024:  # Меньше 1MB
                    st.warning(f"⚠️ Файл слишком мал ({total_size} байт). Возможно, это не модель.")
                
                # Создаем прогресс-бар
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Сохраняем файл
                downloaded = 0
                with open(MODEL_FILENAME, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = downloaded / total_size
                                progress_bar.progress(min(progress, 1.0))
                                status_text.text(f"Загружено: {downloaded/(1024*1024):.1f} MB")
                
                progress_bar.empty()
                status_text.empty()
                
                # Проверяем размер скачанного файла
                if os.path.exists(MODEL_FILENAME):
                    file_size = os.path.getsize(MODEL_FILENAME)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    if file_size > 1024 * 1024:  # > 1MB
                        st.success(f"✅ Модель скачана! Размер: {file_size_mb:.1f} MB")
                    elif file_size > 0:
                        st.warning(f"⚠️ Файл скачан ({file_size_mb:.2f} MB), но может быть слишком мал")
                    else:
                        st.error("❌ Скачан пустой файл")
                        os.remove(MODEL_FILENAME)
                        return None, device
        
        # Загружаем модель в память
        if os.path.exists(MODEL_FILENAME):
            file_size = os.path.getsize(MODEL_FILENAME)
            if file_size == 0:
                st.error("❌ Файл модели пустой")
                return None, device
                
            with st.spinner("🔄 Загрузка модели в память..."):
                model = torch.load(
                    MODEL_FILENAME, 
                    map_location=device,
                    weights_only=False
                )
                model.eval()
                
                file_size_mb = file_size / (1024 * 1024)
                st.success(f"✅ Модель загружена! ({file_size_mb:.1f} MB)")
                return model, device
        else:
            st.error("❌ Файл модели не найден")
            return None, device
        
    except torch.serialization.pickle.UnpicklingError as e:
        st.error(f"❌ Ошибка загрузки pickle файла: {e}")
        st.info("Это означает, что скачанный файл не является валидной моделью PyTorch.")
        return None, device
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {str(e)}")
        return None, device

# Загружаем модель при старте
if 'model' not in st.session_state:
    with st.spinner("Инициализация приложения..."):
        model, device = load_model()
        if model:
            st.session_state.model = model
            st.session_state.device = device
        else:
            st.error("Не удалось загрузить модель")

# ========== БОКОВАЯ ПАНЕЛЬ ==========
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    
    # Раздел: Загрузка модели
    st.markdown("### 🌀 Загрузка модели")
    if 'model' in st.session_state:
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.success(f"Модель загружена на: **{device_name}**")
        
        # Кнопка для перезагрузки модели
        if st.button("🔄 Перезагрузить модель"):
            if os.path.exists(MODEL_FILENAME):
                os.remove(MODEL_FILENAME)
            st.session_state.pop('model', None)
            st.session_state.pop('device', None)
            st.rerun()
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
