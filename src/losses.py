import tensorflow.keras.backend as K
import numpy as np 
import tensorflow as tf
################################################################################
# Objects
################################################################################
class KerasObject:
    def __init__(self, name=None):
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name):
        self._name = name

class Metric(KerasObject):
    pass

class Loss(KerasObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)

class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{}({})'.format(multiplier, loss.__name__)
        else:
            name = '{}{}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt, pr):
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt, pr):
        return self.l1(gt, pr) + self.l2(gt, pr)

def gather_channels(*xs):
    return xs

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))

    return K.mean(loss)

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss_a = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss_b = - (1 - y_true) * ((1 - alpha) * K.pow((y_pred), gamma) * K.log(1 - y_pred))
    
    return K.mean(loss_a + loss_b)

class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name="focal_loss")
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        return categorical_focal_loss(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma
        )


class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        return binary_focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)


def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x


def dice_coefficient(y_true, y_pred, beta=1.0, class_weights=1., smooth=1e-5, threshold=None):
    # print(y_pred)
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)
    # print("Score, wo avg: " + str(score))
    score = average(score, class_weights)
    # print("Score: " + str(score))

    return score

class DiceLoss(Loss):
    def __init__(self, beta=1.0, class_weights=None, smooth=1e-5):
        super().__init__(name="dice_loss")
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        # print(y_pred)
        return 1.0 - dice_coefficient(
            y_true,
            y_pred,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )

