import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Ładowanie danych
(data_train, labels_train), (data_test, labels_test) = mnist.load_data()

# Przeskalowanie i zmiana kształtu danych
data_train = data_train.reshape(data_train.shape[0], 28, 28, 1).astype('float32') / 255
data_test = data_test.reshape(data_test.shape[0], 28, 28, 1).astype('float32') / 255

# Konwersja etykiet na kategorie
labels_train = keras.utils.to_categorical(labels_train, 10)
labels_test = keras.utils.to_categorical(labels_test, 10)

# Zmiana etykiet w 50% danych uczących
indices = np.random.choice(np.arange(len(labels_train)), len(labels_train) // 2, replace=False)
wrong_labels = np.random.randint(0, 10, len(indices))
labels_train[indices] = keras.utils.to_categorical(wrong_labels, 10)

# Funkcja tworząca model MLP
def create_mlp_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Funkcja tworząca model CNN
def create_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

mlp_model = create_mlp_model()
cnn_model = create_cnn_model()

# Early stopping and model checkpoint
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

mlp_history = mlp_model.fit(data_train, labels_train, epochs=20, validation_data=(data_test, labels_test), callbacks=callbacks)
cnn_history = cnn_model.fit(data_train, labels_train, epochs=20, validation_data=(data_test, labels_test), callbacks=callbacks)


# Funkcja wyświetlająca predykcje
def display_predictions(model, images, true_labels, class_names):
    preds = model.predict(images)
    preds_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(true_labels, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(f"Pred: {class_names[preds_classes[i]]}\nTrue: {class_names[true_classes[i]]}")
    plt.show()

# Wybór losowych obrazów do testowania predykcji
indices = np.random.choice(range(len(data_test)), 25, replace=False)
sample_images = data_test[indices]
sample_labels = labels_test[indices]
class_names = [str(i) for i in range(10)]  # Dla MNIST
# Wybierz losowo 25 obrazów z zestawu testowego
indices = np.random.choice(range(len(data_test)), 25, replace=False)
sample_images = data_test[indices]
sample_labels = labels_test[indices]
class_names = [str(i) for i in range(10)]

# Wyświetl predykcje dla MLP
display_predictions(mlp_model, sample_images, sample_labels, class_names)

# Wyświetl predykcje dla CNN
display_predictions(cnn_model, sample_images, sample_labels, class_names)

# Wykresy dokładności
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'], label='MLP Train Accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='MLP Validation Accuracy')
plt.title('MLP Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Wykresy strat
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['loss'], label='MLP Train Loss')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')
plt.title('MLP Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='CNN Train Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

