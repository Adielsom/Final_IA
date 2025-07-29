# Segmentação de Lesões de Pele com U-Net)

Este projeto implementa um pipeline completo para a segmentação automática de lesões de pele em imagens de dermatoscopia utilizando uma arquitetura de rede neural convolucional baseada na U-Net, com aprimoramentos como Blocos Residuais e Attention Gates.

Visão Geral do Projeto
O objetivo principal deste trabalho é desenvolver um modelo capaz de identificar e delinear com precisão a área de lesões de pele em imagens médicas. A segmentação precisa é um passo crucial para sistemas de diagnóstico auxiliado por computador na dermatologia, ajudando médicos na avaliação e acompanhamento de pintas e outras lesões, incluindo potenciais melanomas.
Utilizamos a arquitetura U-Net, conhecida por seu excelente desempenho em tarefas de segmentação biomédica, e aprimoramos sua capacidade de aprendizado e foco em regiões relevantes através da incorporação de Blocos Residuais e Attention Gates.

Arquitetura do Modelo: U-Net Aprimorada
A arquitetura central do projeto é uma U-Net, uma CNN com uma estrutura simétrica em forma de "U", ideal para segmentação de imagens.

Encoder (Caminho de Contração): Sequência de blocos convolucionais e operações de max-pooling que reduzem a resolução espacial e aumentam a profundidade (número de filtros), capturando características contextuais.
Decoder (Caminho de Expansão): Sequência de operações de up-sampling e blocos convolucionais que gradualmente restauram a resolução espacial.
Skip Connections: Conexões diretas entre camadas correspondentes do encoder e do decoder. Elas permitem que informações espaciais de alta resolução do início da rede sejam combinadas com informações contextuais do final do encoder, melhorando a precisão dos contornos segmentados.
Aprimoramentos:

Blocos Residuais: Substituem os blocos convolucionais padrão. Adicionam uma "conexão de atalho" (shortcut) que permite que o gradiente flua mais facilmente durante o treinamento, facilitando o treinamento de redes mais profundas e mitigando o problema do gradiente evanescente.
Attention Gates: Inseridos nas skip connections. Aprendem a focar (atenuar ou realçar) as características relevantes provenientes do encoder antes de concatená-las com as características do decoder, ajudando o modelo a concentrar seus recursos computacionais nas regiões de interesse.
A combinação desses aprimoramentos busca tornar o modelo mais robusto, eficiente e preciso na tarefa de segmentação.

Dataset
O projeto utiliza o dataset ISIC 2018 Challenge Task 1 - Lesion Boundary Segmentation.

Origem: Parte do desafio anual da International Skin Imaging Collaboration (ISIC).
Conteúdo: Consiste em pares de imagens de dermatoscopia e suas respectivas máscaras de segmentação (Ground Truth) anotadas por especialistas.
Tamanho: O dataset original contém 2594 pares de imagem/máscara. Para o treinamento, foi dividido em:
Treino: 2075 amostras (aprox. 80%)
Validação: 519 amostras (aprox. 20%)
Este dataset é um padrão da indústria para validação de algoritmos de segmentação de lesões de pele, garantindo que o modelo seja treinado e avaliado em dados realistas e de alta qualidade.

Pré-processamento e Carregamento de Dados
O pipeline de dados inclui:

Download: O dataset é baixado automaticamente usando o KaggleHub.
Carregamento: Imagens e máscaras são carregadas a partir dos caminhos de arquivo.
Redimensionamento: Todas as imagens e máscaras são redimensionadas para o tamanho IMG_SIZE (128x128 pixels) para garantir consistência na entrada do modelo.
Normalização: Os valores dos pixels são normalizados para o intervalo [0, 1] dividindo por 255.0.
Divisão: Os dados são divididos em conjuntos de treino e validação usando train_test_split do scikit-learn.
(Opcional) Aumento de Dados: O código inclui uma função augment_data para aplicar transformações como rotação, flips horizontal/vertical e ajuste de brilho no conjunto de treino, embora não esteja ativada por padrão na função create_dataset e main no estado atual do notebook. A aplicação de aumento de dados geralmente melhora a robustez do modelo.
Métricas e Função de Perda
Para treinar e avaliar o modelo, foram utilizadas as seguintes métricas e função de perda:

Dice Coefficient: Mede a similaridade entre a área predita e a área real da lesão. Valor entre 0 e 1, onde 1 é a sobreposição perfeita. É uma métrica robusta para datasets com desbalanceamento de classes.
IoU Score (Intersection over Union): Similar ao Dice, mede a proporção da área de sobreposição em relação à área total combinada. Também varia de 0 a 1.
Binary Accuracy: Proporção de pixels corretamente classificados (lesão ou fundo). Útil, mas menos indicativa que Dice/IoU em casos de classes desbalanceadas.
Combined Loss: A função de perda utilizada para otimizar o modelo é a soma da Dice Loss (1 - Dice Coefficient) e da Binary Crossentropy. Esta combinação ajuda o modelo a focar tanto na sobreposição da área da lesão quanto na classificação correta de cada pixel individual.
Treinamento
O modelo é treinado utilizando:

Otimizador: Adam
Taxa de Aprendizado: LEARNING_RATE = 1e-4
Batch Size: BATCH_SIZE = 8
Épocas: EPOCHS = 5 (O treinamento foi configurado para 5 épocas no notebook, mas callbacks como Early Stopping podem pará-lo antes ou o modelo pode continuar aprendendo além disso dependendo da configuração e dados).
Callbacks:
EarlyStopping: Monitora o val_dice_coefficient e para o treinamento se não houver melhora por 10 épocas (patience=10), restaurando os melhores pesos.
ReduceLROnPlateau: Monitora a val_loss e reduz a taxa de aprendizado pela metade se não houver melhora por 5 épocas (patience=5).
ModelCheckpoint: Salva o modelo com o melhor val_dice_coefficient.
Resultados
Após 5 épocas de treinamento, o modelo obteve os seguintes resultados no conjunto de validação:

Loss: 0.3594
Dice Coefficient: 0.8481
IoU Score: 0.7380
Binary Accuracy: 0.9411
Um coeficiente Dice de aproximadamente 0.85 e um IoU de 0.74 são considerados bons resultados para tarefas de segmentação médica e indicam que o modelo é capaz de segmentar as lesões com uma precisão considerável.

As visualizações das predições (mostradas no notebook) confirmam que o modelo consegue gerar máscaras que se sobrepõem bem às lesões nas imagens originais.
