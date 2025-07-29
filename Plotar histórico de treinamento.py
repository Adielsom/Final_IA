def plot_training_history(history):
    """Plotar histórico de treinamento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Validation')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Dice Coefficient
    axes[0, 1].plot(history.history['dice_coefficient'], label='Train')
    axes[0, 1].plot(history.history['val_dice_coefficient'], label='Validation')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()

    # IoU Score
    axes[1, 0].plot(history.history['iou_score'], label='Train')
    axes[1, 0].plot(history.history['val_iou_score'], label='Validation')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()

    # Learning Rate (se disponível)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()


----------------------

import cv2
import numpy as np
from tqdm import tqdm # Para uma barra de progresso visual

def load_data_from_paths(image_paths, mask_paths, img_size=(128, 128)):
    """Carrega imagens e máscaras a partir de listas de caminhos."""
    images = []
    masks = []

    # A barra de progresso (tqdm) é útil para datasets grandes
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Carregando dados"):
        # Carregar imagem
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalizar

        # Carregar máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0  # Normalizar
        mask = np.expand_dims(mask, axis=-1) # Adicionar dimensão de canal -> (128, 128, 1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)
