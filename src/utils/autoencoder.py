import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model
import numpy as np
import json

from utils.net2net import net2wider, net2deeper


class PartCoder(Layer):
    def __init__(self, output_dim=2, hidden_dims=None, l1=0.01, l2=0.01):
        super(PartCoder, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.layers = []
        for i, dim in enumerate(hidden_dims):
            layer = Dense(
                units=dim,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2)
            )
            self.layers.append(layer)

        # Final, adding output_layer (latent/reconstruction layer)
        self.layers.append(Dense(
            units=output_dim,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2)
        ))

    def wider(self, added_size=1, pos_layer=None):
        layers_size = len(self.layers)
        if pos_layer is None:
            pos_layer = layers_size - 2
        elif pos_layer >= layers_size - 1:
            raise ValueError(
                f"pos_layer is expected less than length of layers. pos_layer in [0, layers_size-2]")

        weights, bias = self.layers[pos_layer].get_weights()
        weights_next_layer, _ = self.layers[pos_layer + 1].get_weights()

        pass
        # new_weights, new_bias, new_weights_next_layer = net2wider(weights, bias, weights_next_layer)

    def deeper(self, pos_layer=None):
        layers_size = len(self.layers)
        if pos_layer is None:
            pos_layer = max(layers_size - 2, 0)
        elif pos_layer >= layers_size - 1:
            raise ValueError(
                f"pos_layer is expected less than length of layers. pos_layer in [0, layers_size-2]")

        weights, bias = self.layers[pos_layer].get_weights()
        new_weights, new_bias = net2deeper(weights)
        src_units = des_units = weights.shape[1]
        layer = Dense(
            units=des_units,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2),
            # kernel_initializer='glorot_uniform'
        )
        layer.build(input_shape=(src_units, des_units))
        layer.set_weights([new_weights, new_bias])

        self.layers.insert(pos_layer + 1, layer)

    def call(self, inputs):
        z = inputs
        for layer in self.layers:
            z = layer(z)

        return z

    def info(self, show_weight=False, show_config=False):
        print(f"{self.name}\n----------")
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}\n\t Name={layer.name}\n\t Shape ={layer.get_weights()[0].shape}")
            if show_weight:
                print(f"\t Weight= {layer.get_weights()}")
            if show_config:
                print(f"Config: {json.dumps(layer.get_config(), sort_keys=True, indent=4)}")


class Autoencoder(Model):
    def __init__(self, input_dim, embedding_dim, hidden_dims=None, v1=0.01, v2=0.01):
        super(Autoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512]

        self.encoder = PartCoder(output_dim=embedding_dim, hidden_dims=hidden_dims, l1=v1, l2=v2)
        self.decoder = PartCoder(output_dim=input_dim, hidden_dims=hidden_dims[::-1], l1=v1, l2=v2)

    def call(self, inputs):
        Y = self.encoder(inputs)
        X_hat = self.decoder(Y)
        return X_hat, Y

    def get_embedding(self, inputs):
        return self.encoder(inputs)

    def get_reconstruction(self, inputs):
        return self.decoder(self.encoder(inputs))


if __name__ == "__main__":
    print("\n#######\nEncoder")
    # Suppose: 4 -> 3-> 5 -> 2
    encoder = PartCoder(output_dim=2, hidden_dims=[3, 5])
    x = tf.ones((3, 4))
    y = encoder(x)
    # print("y=", y)
    # encoder.info(show_weight=True, show_config=False)
    encoder.deeper()
    y = encoder(x)
    # print("y=", y)
    print("After deeper")
    encoder.info(show_weight=True, show_config=False)

    # ----------- Decoder -----------
    print("\n####\nDecoder")
    # Suppose: 2 -> 5 -> 3 -> 4

    decoder = PartCoder(output_dim=4, hidden_dims=[5, 3])
    x = tf.ones((3, 2))
    y = decoder(x)
    # print("y=", y)
    # encoder.info(show_weight=True, show_config=False)
    decoder.deeper()
    y = decoder(x)
    # print("y=", y)
    print("After deeper")
    decoder.info(show_weight=True, show_config=False)
