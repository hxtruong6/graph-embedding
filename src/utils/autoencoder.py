import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model


class Encoder(Layer):
    def __init__(self, embedding_dim=2, hidden_dims=None, v1=0.01, v2=0.01):
        super(Encoder, self).__init__()

        self.layers = []
        for i, dim in enumerate(hidden_dims):
            layer = Dense(
                units=dim,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=v1, l2=v2)
            )
            self.layers.append(layer)

        self.output_layer = Dense(
            units=embedding_dim,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=v1, l2=v2)
        )

    def call(self, inputs):
        z = inputs
        for i in range(len(self.layers)):
            z = self.layers[i](z)

        return self.output_layer(z)


class Decoder(Layer):
    def __init__(self, input_dim, hidden_dims=None, v1=0.01, v2=0.02):
        super(Decoder, self).__init__()

        self.layers = []
        for i, dim in enumerate(hidden_dims):
            layer = Dense(
                units=dim,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=v1, l2=v2)
            )
            self.layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(
            units=input_dim,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=v1, l2=v2)
        )

    def call(self, Y):
        '''

        :param Y: embedding layer
        :return:
        '''
        z = Y
        for i in range(len(self.layers)):
            z = self.layers[i](z)

        return self.output_layer(z)


class Autoencoder(Model):
    def __init__(self, input_dim, embedding_dim, hidden_dims=None, v1=0.01, v2=0.01):
        super(Autoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512]

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            v1=v1,
            v2=v2
        )

        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims[::-1],
            v1=v1,
            v2=v2
        )

    def call(self, inputs):
        Y = self.encoder(inputs)
        X_hat = self.decoder(Y)
        return X_hat, Y

    def get_embedding(self, inputs):
        return self.encoder(inputs)

    def get_reconstruction(self, inputs):
        return self.decoder(self.encoder(inputs))
