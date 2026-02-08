import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def predict_with_pixel_area(model, image_path, device, pixel_area_m2=0.000451, threshold=0.5, show_result=False):
    """
    Оценивает площадь зданий используя ПЛОЩАДЬ ОДНОГО ПИКСЕЛЯ
    
    Args:
        model: обученная модель сегментации
        image_path: путь к изображению
        pixel_area_m2: площадь одного пикселя ОРИГИНАЛЬНОГО изображения в м²
        threshold: порог для бинарной сегментации
        
    Returns:
        dict с результатами анализа
    """
    model.eval()
    
    # Загрузка изображения
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Ошибка: не удалось загрузить изображение {image_path}")
        return None
    
    # Сохраняем оригинальные размеры
    original_height, original_width = image.shape[:2]
    
    # Преобразование в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Изменение размера до 256x256 (как обучалась модель)
    image_resized = cv2.resize(image_rgb, (256, 256))
    
    # Нормализация
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Преобразование в тензор
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Предсказание
        output = model(image_tensor)  # [1, 1, 256, 256]
        pred_probs = torch.sigmoid(output)
        pred_mask_small = (pred_probs > threshold).squeeze(0).squeeze(0).cpu()  # [256, 256], bool
    
    # МАСШТАБИРОВАНИЕ МАСКИ ДО ОРИГИНАЛЬНОГО РАЗМЕРА (ДЛЯ STREAMLIT)
    pred_mask_original = cv2.resize(pred_mask_small.numpy().astype(np.uint8), 
                                   (original_width, original_height), 
                                   interpolation=cv2.INTER_NEAREST)
    
    # РАСЧЕТЫ ПЛОЩАДИ
    building_mask_small = pred_mask_small.numpy().astype(np.uint8)
    pixels_count_small = np.sum(building_mask_small)
    
    scale_x = original_width / 256
    scale_y = original_height / 256
    area_scale_factor = scale_x * scale_y
    
    pixels_count_original = pixels_count_small * area_scale_factor
    total_area_m2 = pixels_count_original * pixel_area_m2
    
    # ПОДСЧЕТ ЗДАНИЙ НА ОРИГИНАЛЬНОЙ МАСКЕ
    if np.sum(pred_mask_original == 1) > 0:
        num_labels, _ = cv2.connectedComponents(pred_mask_original)
        num_buildings = num_labels - 1
    else:
        num_buildings = 0
    
    # Общая площадь всего изображения
    total_image_area_m2 = original_width * original_height * pixel_area_m2
    
    results = {
        'image_rgb': image_rgb,  # Оригинальное изображение (оригинальный размер)
        'pred_mask_small': building_mask_small,  # Маска 256x256
        'pred_mask_original': pred_mask_original,  # Маска оригинального размера
        'total_area_m2': total_area_m2,
        'total_image_area_m2': total_image_area_m2,
        'num_buildings': num_buildings,
        'original_width': original_width,
        'original_height': original_height,
        'pixels_count_small': pixels_count_small,
        'pixels_count_original': pixels_count_original,
        'pixel_area_m2': pixel_area_m2,
        'area_scale_factor': area_scale_factor
    }

    if show_result:
        show_simple_results(results, image_path)
        return None
    
    return results

def show_simple_results(results, image_path):
    """
    Краткая визуализация результатов
    """
    # Создаем цветную маску для визуализации (оригинальный размер)
    color_mask = np.zeros((results['original_height'], results['original_width'], 3), dtype=np.uint8)
    color_mask[results['pred_mask_original'] == 1] = [255, 0, 0]  # Используем pred_mask_original!
    
    # Наложение маски на оригинал
    overlay = results['image_rgb'].copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1-alpha, color_mask, alpha, 0)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Оригинальное изображение
    axes[0].imshow(results['image_rgb'])
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    # Результат сегментации
    axes[1].imshow(overlay)
    axes[1].set_title(f'Сегментация ({results["num_buildings"]} зданий)')
    axes[1].axis('off')
    
    plt.suptitle(f'Результаты анализа: {os.path.basename(image_path)}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Краткая сводка
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Размер изображения: {results['original_width']}×{results['original_height']} пикс.")
    print(f"   Площадь пикселя: {results['pixel_area_m2']:.4f} м²")
    print(f"   Общая площадь территории: {results['total_image_area_m2']:,.1f} м²")
    print(f"   Обнаружено зданий: {results['num_buildings']}")
    print(f"   Площадь зданий: {results['total_area_m2']:,.1f} м²")
    
    return results