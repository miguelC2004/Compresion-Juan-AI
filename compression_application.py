import tensorflow as tf
from tensorflow.keras import layers, models

def build_compression_model(input_shape):
    # Codificador
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded_output = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decodificador
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_output)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Modelo completo
    model = models.Model(encoder_input, decoder_output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
