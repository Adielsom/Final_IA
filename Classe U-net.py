class UNetModel:
      def __init__(self, input_shape, num_classes=1):
        # Construtor da classe UNetModel.
        # Ele inicializa a forma esperada das imagens de entrada (input_shape)
        # e o número de classes que o modelo deve segmentar (num_classes).
        # Para segmentação binária (lesão vs. não lesão), num_classes será 1 (saída sigmoid).

        self.input_shape = input_shape
        self.num_classes = num_classes

      def conv_block(self, inputs, num_filters, kernel_size=3, padding="same",
                    use_batch_norm=True, dropout_rate=0.0):
          """
          Bloco convolucional melhorado com BatchNormalization e Dropout, unidade
          fundamental do encoder e decoder da U-Net
          """

           # Primeira camada convolucional
          conv = Conv2D(num_filters, kernel_size, padding=padding)(inputs)
          if use_batch_norm:
              conv = BatchNormalization()(conv) # Normaliza as ativações, acelerando o treinamento
          conv = Activation('relu')(conv) # Função de ativação ReLU (Rectified Linear Unit)

          # Segunda camada convolucional
          conv = Conv2D(num_filters, kernel_size, padding=padding)(conv)
          if use_batch_norm:
              conv = BatchNormalization()(conv)
          conv = Activation('relu')(conv)

          if dropout_rate > 0:
              conv = Dropout(dropout_rate)(conv) # Aplica Dropout, ajuda a prevenir overfitting

          return conv

      def attention_gate(self, gate, skip_connection, num_filters):
          """
          Implementação do Attention Gate para U-Net
          Ajuda o modelo a focar nas regiões mais relevantes
          """
         # Transforma o sinal do decoder (gate) para o mesmo número de filtros
          gate_conv = Conv2D(num_filters, 1, padding="same")(gate)
          gate_conv = BatchNormalization()(gate_conv)

           # Transforma a conexão de atalho (skip_connection) para o mesmo número de filtros
          skip_conv = Conv2D(num_filters, 1, padding="same")(skip_connection)
          skip_conv = BatchNormalization()(skip_conv)

          # Soma os dois sinais transformados e aplica ReLU
          merged = Add()([gate_conv, skip_conv])
          merged = Activation('relu')(merged)

           # Calcula o mapa de atenção (weights)
          attention = Conv2D(1, 1, padding="same")(merged)
          attention = Activation('sigmoid')(attention) # Sigmoid para gerar pesos entre 0 e 1

           # Aplica o mapa de atenção à conexão de atalho
          attended_skip = multiply([skip_connection, attention])

          return attended_skip

      def residual_block(self, inputs, num_filters):
          """
          Bloco residual para melhorar o fluxo de gradiente
          """
          # Primeira convolução no caminho principal
          conv1 = Conv2D(num_filters, 3, padding="same")(inputs)
          conv1 = BatchNormalization()(conv1)
          conv1 = Activation('relu')(conv1)

          # Segunda convolução no caminho principal
          conv2 = Conv2D(num_filters, 3, padding="same")(conv1)
          conv2 = BatchNormalization()(conv2)

          # Conexão de atalho (shortcut connection):
          # Se o número de filtros da entrada for diferente do número de filtros de saída do bloco,
          # uma convolução 1x1 é usada para ajustar as dimensões. Caso contrário, a entrada é usada diretamente.
          shortcut = Conv2D(num_filters, 1, padding="same")(inputs) if inputs.shape[-1] != num_filters else inputs
          shortcut = BatchNormalization()(shortcut) # Normalização também na atalho

          # Adicionar conexão residual
          merged = Add()([conv2, shortcut])
          merged = Activation('relu')(merged) # Ativação final do bloco residual

          return merged

      def build_model(self, use_attention=True, use_residual=True):
          """
          Constrói a arquitetura U-Net melhorada
          """
          inputs = Input(self.input_shape)  # Define a camada de entrada do modelo

          # --- Encoder (Caminho de Contração) ---
          # A resolução da imagem é progressivamente reduzida enquanto o número de filtros (características) aumenta.


           # Bloco 1 (entrada para 64 filtros)
           # Usa bloco residual ou convolucional padrão
          conv1 = self.residual_block(inputs, 64) if use_residual else self.conv_block(inputs, 64)
          pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # Reduz a resolução pela metade

          # Bloco 2 (pool1 para 128 filtros)
          conv2 = self.residual_block(pool1, 128) if use_residual else self.conv_block(pool1, 128)
          pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

          # Bloco 3 (pool2 para 256 filtros)
          conv3 = self.residual_block(pool2, 256) if use_residual else self.conv_block(pool2, 256)
          pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

          # Bloco 4 (pool3 para 512 filtros)
          conv4 = self.residual_block(pool3, 512) if use_residual else self.conv_block(pool3, 512)
          pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

          # --- Bridge (Bottleneck) ---
          # A camada mais profunda da rede, onde a informação é mais comprimida e contextual.
          bridge = self.residual_block(pool4, 1024) if use_residual else self.conv_block(pool4, 1024, dropout_rate=0.5)

          # --- Decoder (Caminho de Expansão) ---
          # A resolução é gradualmente restaurada, combinando informações de alto nível
          # (do bridge) com informações de baixo nível (das skip connections).


          # Bloco de expansão 1 (up1 para 512 filtros)
          # upsampling do bridge e concatenação com conv4 (ou conv4_att)
          up1 = Conv2DTranspose(512, 2, strides=(2, 2), padding="same")(bridge)
          if use_attention:
            # Aplica Attention Gate na skip connection conv4
              conv4_att = self.attention_gate(up1, conv4, 512)
              merge1 = concatenate([up1, conv4_att], axis=3) # Concatena upsampled com a skip connection atenuada
          else:
              merge1 = concatenate([up1, conv4], axis=3) # Concatenação padrão
          conv5 = self.residual_block(merge1, 512) if use_residual else self.conv_block(merge1, 512)

          # Bloco de expansão 2 (up2 para 256 filtros)
          # upsampling de conv5 e concatenação com conv3 (ou conv3_att)
          up2 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(conv5)
          if use_attention:
              conv3_att = self.attention_gate(up2, conv3, 256)
              merge2 = concatenate([up2, conv3_att], axis=3)
          else:
              merge2 = concatenate([up2, conv3], axis=3)
          conv6 = self.residual_block(merge2, 256) if use_residual else self.conv_block(merge2, 256)

          # Bloco de expansão 3 (up3 para 128 filtros)
          # upsampling de conv6 e concatenação com conv2 (ou conv2_att)
          up3 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(conv6)
          if use_attention:
              conv2_att = self.attention_gate(up3, conv2, 128)
              merge3 = concatenate([up3, conv2_att], axis=3)
          else:
              merge3 = concatenate([up3, conv2], axis=3)
          conv7 = self.residual_block(merge3, 128) if use_residual else self.conv_block(merge3, 128)

          # Bloco de expansão 4 (up4 para 64 filtros)
          # upsampling de conv7 e concatenação com conv1 (ou conv1_att)
          up4 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(conv7)
          if use_attention:
              conv1_att = self.attention_gate(up4, conv1, 64)
              merge4 = concatenate([up4, conv1_att], axis=3)
          else:
              merge4 = concatenate([up4, conv1], axis=3)
          conv8 = self.residual_block(merge4, 64) if use_residual else self.conv_block(merge4, 64)

             # Camada de saída
             # Uma convolução 1x1 final mapeia as características para o número de classes desejado.
             # 'sigmoid' é usada para segmentação binária (0 ou 1).
          outputs = Conv2D(self.num_classes, 1, activation="sigmoid")(conv8)

         # Cria o modelo Keras com as entradas e saídas definidas
          model = Model(inputs=[inputs], outputs=[outputs])
          return model
