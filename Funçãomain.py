def main():
    """Função principal para executar o pipeline completo"""
    print("=== Iniciando Pipeline de Segmentação de Lesões de Pele ===")

    # 1. Configurar carregador de dados
    data_loader = DataLoader(img_size=IMG_SIZE)

    # 2. Baixar dataset (descomente se necessário)
    dataset_path = data_loader.download_dataset()
    if dataset_path is None:
      print("Erro: Não foi possível baixar o dataset")
      return

    # Caminho dos dados.
    IMAGE_PATH = '/root/.cache/kagglehub/datasets/tschandl/isic2018-challenge-task1-data-segmentation/versions/1/ISIC2018_Task1-2_Training_Input/'
    MASK_PATH = '/root/.cache/kagglehub/datasets/tschandl/isic2018-challenge-task1-data-segmentation/versions/1/ISIC2018_Task1_Training_GroundTruth/'


    # 3. Configurar caminhos (descomente e ajuste conforme necessário)
    image_files, mask_files = data_loader.setup_paths(dataset_path)

    # 4. Dividir dados
    X_train_paths, X_val_paths, y_train_paths, y_val_paths = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
    )

    print(f"Usando {len(image_files)} amostras do dataset real.")
    print(f"Conjunto de treino: {len(X_train_paths)} imagens")
    print(f"Conjunto de validação: {len(X_val_paths)} imagens")

    print("\nCarregando dados de treino para a memória...")
    X_train, y_train = load_data_from_paths(X_train_paths, y_train_paths, img_size=IMG_SIZE)

    print("\nCarregando dados de validação para a memória...")
    X_val, y_val = load_data_from_paths(X_val_paths, y_val_paths, img_size=IMG_SIZE)
    # -------------------------------------------------------------------------

    print(f"\nFormatos dos dados carregados:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # 5. Criar modelo
    unet = UNetModel(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = unet.build_model(use_attention=True, use_residual=True)

    # 6. Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=MetricsAndLoss.combined_loss,
        metrics=[
            MetricsAndLoss.dice_coefficient,
            MetricsAndLoss.iou_score,
            'binary_accuracy'
        ]
    )

    print("Arquitetura do modelo:")
    model.summary()

    # 7. Configurar callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=10,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            'best_unet_model.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True
        )
    ]

    # 8. Treinar modelo
    print("\n=== Iniciando Treinamento ===")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 9. Plotar resultados
    plot_training_history(history)

    # 10. Avaliar modelo
    print("\n=== Avaliação Final ===")
    results = model.evaluate(X_val, y_val, verbose=0)
    print(f"Loss: {results[0]:.4f}")
    print(f"Dice Coefficient: {results[1]:.4f}")
    print(f"IoU Score: {results[2]:.4f}")
    print(f"Binary Accuracy: {results[3]:.4f}")

    # 11. Visualizar predições
    visualize_predictions(model, X_val, y_val, num_samples=5)

    print("\n=== Pipeline Concluído ===")
    return model, history

