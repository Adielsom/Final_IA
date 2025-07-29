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
