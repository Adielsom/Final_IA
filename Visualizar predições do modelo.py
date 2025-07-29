def visualize_predictions(model, X_val, y_val, num_samples=5):
    """Visualizar predições do modelo"""
    predictions = model.predict(X_val[:num_samples])

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))

    for i in range(num_samples):
        # Imagem original
        axes[i, 0].imshow(X_val[i])
        axes[i, 0].set_title('Imagem Original')
        axes[i, 0].axis('off')

        # Máscara verdadeira
        axes[i, 1].imshow(y_val[i][:,:,0], cmap='gray')
        axes[i, 1].set_title('Máscara Verdadeira')
        axes[i, 1].axis('off')

        # Predição
        pred_binary = (predictions[i][:,:,0] > 0.5).astype(np.float32)
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title('Predição (Threshold=0.5)')
        axes[i, 2].axis('off')

        # Sobreposição
        overlay = X_val[i].copy()
        overlay[:,:,0] = np.maximum(overlay[:,:,0], pred_binary * 0.7)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Sobreposição')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()
