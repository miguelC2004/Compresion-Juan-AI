import tensorflow as tf

class CompressionModel(tf.keras.Model):
    def __init__(self, input_shape=(None, None, 1)):
        super(CompressionModel, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=input_shape)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
