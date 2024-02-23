import tensorflow as tf
from compression_model import CompressionModel

# Cargar los datos de imágenes (ajustar según tus necesidades)
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# Normalización de las imágenes después de cargar los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

# Agregar una dimensión para el canal de color
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

# Crear una instancia del modelo
model = CompressionModel(input_shape=(28, 28, 1))  # Ajustar dimensiones según tus imágenes

# Compilar el modelo con métricas adecuadas
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_images, epochs=10, batch_size=128, shuffle=True, validation_data=(test_images, test_images))

# Evaluar el modelo
loss, accuracy = model.evaluate(test_images, test_images)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Guardar el modelo entrenado
model.save('compression_model.h5')
