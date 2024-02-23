import tensorflow as tf
from compression_model import CompressionModel

# No supe cómo hacer para cargar los datos de imágenes
# Supongamos que tienes un conjunto de datos `train_images` y `test_images`

# Normalización de las imágenes
train_images = train_images / 255.0
test_images = test_images / 255.0

# Agregar una dimensión para el canal de color
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

# Crear una instancia del modelo
model = CompressionModel()

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(train_images, train_images, epochs=10, batch_size=128, shuffle=True, validation_data=(test_images, test_images))

# Evaluar el modelo
loss = model.evaluate(test_images, test_images)
print("Loss:", loss)

# Guardar el modelo entrenado
model.save('compression_model.h5')
