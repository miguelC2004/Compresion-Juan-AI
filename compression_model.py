import tensorflow as tf
from compression_model import build_compression_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuración
batch_size = 32
image_size = (256, 256)
input_shape = image_size + (3,)
epochs = 10
data_dir = 'path/to/your/dataset'

# Cargando y preparando el conjunto de datos
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,  # Sin etiquetas, solo imágenes
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    subset='validation'
)

# Construir modelo
model = build_compression_model(input_shape)

# Entrenamiento
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Guardar modelo
model.save('compression_model.h5')

print("Modelo entrenado y guardado con éxito.")
