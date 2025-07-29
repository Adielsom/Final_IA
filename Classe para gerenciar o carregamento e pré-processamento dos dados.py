class DataLoader:
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def download_dataset(self):
        """Download do dataset ISIC do Kaggle"""
        try:
            path = kagglehub.dataset_download("tschandl/isic2018-challenge-task1-data-segmentation")
            print(f"Dataset baixado em: {path}")
            return path
        except Exception as e:
            print(f"Erro ao baixar dataset: {e}")
            return None

    def setup_paths(self, dataset_path):
        """Configurar caminhos para imagens e máscaras"""
        image_path = os.path.join(dataset_path, 'ISIC2018_Task1-2_Training_Input')
        mask_path = os.path.join(dataset_path, 'ISIC2018_Task1_Training_GroundTruth')

        image_files = sorted(glob.glob(os.path.join(image_path, '*.jpg')))
        mask_files = sorted(glob.glob(os.path.join(mask_path, '*.png')))

        print(f"Total de imagens: {len(image_files)}")
        print(f"Total de máscaras: {len(mask_files)}")

        return image_files, mask_files

    def load_and_preprocess_image(self, img_path, mask_path):
        """Carrega e pré-processa uma imagem e sua máscara"""
        # Carregar imagem
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0

        # Carregar máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        return img, mask

    def augment_data(self, image, mask):
        """Aplicar augmentação de dados"""
        # Rotação aleatória
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
            mask = cv2.warpAffine(mask, matrix, (w, h))

        # Flip horizontal
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Flip vertical
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Ajuste de brilho
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        return image, mask

    def create_dataset(self, image_files, mask_files, num_samples=None, augment=False):
        """Criar dataset com imagens e máscaras"""
        if num_samples is None:
            num_samples = len(image_files)

        images = []
        masks = []

        for i in range(min(num_samples, len(image_files))):
            img, mask = self.load_and_preprocess_image(image_files[i], mask_files[i])

            images.append(img)
            masks.append(mask)

            # Aplicar augmentação se solicitado
            if augment:
                aug_img, aug_mask = self.augment_data(img.copy(), mask.copy())
                images.append(aug_img)
                masks.append(aug_mask)

        return np.array(images), np.array(masks)
