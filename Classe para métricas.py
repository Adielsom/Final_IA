
class MetricsAndLoss:
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Coeficiente Dice"""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_loss(y_true, y_pred):
        """Função de perda Dice"""
        return 1 - MetricsAndLoss.dice_coefficient(y_true, y_pred)

    @staticmethod
    def iou_score(y_true, y_pred, smooth=1e-6):
        """Intersection over Union (IoU)"""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def combined_loss(y_true, y_pred):
        """Perda combinada: Dice + Binary Crossentropy"""
        dice_loss = MetricsAndLoss.dice_loss(y_true, y_pred)
        bce_loss = K.binary_crossentropy(y_true, y_pred)
        return dice_loss + bce_loss
