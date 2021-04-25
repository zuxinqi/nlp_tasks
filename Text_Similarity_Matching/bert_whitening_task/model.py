from bert4keras.models import build_transformer_model
from bert4keras.backend import keras, K
from keras.models import Model

class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)

def create_model(config_path, checkpoint_path):
    # 建立模型
    bert = build_transformer_model(config_path, checkpoint_path)

    encoder_layers, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            encoder_layers.append(output)
            count += 1
        except:
            break

    n_last, outputs = 2, []
    for i in range(n_last):
        outputs.append(GlobalAveragePooling1D()(encoder_layers[-i]))
    output = keras.layers.Average()(outputs)
    # 最后的编码器
    encoder = Model(bert.inputs, output)
    return encoder