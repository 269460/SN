import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

# Load data
df_train = pd.read_csv('input/train.csv')

# Features and labels
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Preprocessing steps for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Prepare data
X = preprocessor.fit_transform(df_train[features])
y = df_train['Survived'].values

# Split data and preserve indices
X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
    X, y, df_train.index, test_size=0.2, random_state=42)

# Define a Keras model (shallow neural network)
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Specify input shape
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluate the model
_, accuracy = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {accuracy:.2f}")

# Make predictions
y_pred = (model.predict(X_val) > 0.5).astype(int)

# Merge predictions with additional passenger information
additional_info = df_train.loc[val_idx, ['Name', 'Age', 'Sex', 'Pclass']]
prediction_results = pd.DataFrame({
    'PassengerId': df_train.loc[val_idx, 'PassengerId'],
    'Name': additional_info['Name'],
    'Age': additional_info['Age'],
    'Sex': additional_info['Sex'],
    'Pclass': additional_info['Pclass'],
    'Survived': y_pred.ravel()
})

# Save predictions to a CSV file
prediction_results.to_csv('predictions.csv', index=False)
print("Predictions have been saved to 'predictions.csv'.")

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.tight_layout()
plt.show()

#
# 1. Importowanie bibliotek
# Kod rozpoczyna się od importowania potrzebnych bibliotek, które umożliwiają manipulację danymi (pandas, numpy), modelowanie (sklearn, tensorflow.keras), oraz wizualizację (matplotlib).
#
# 2. Wczytywanie danych
# pd.read_csv: Wczytuje dane z pliku CSV do obiektu DataFrame (df_train), który zawiera dane treningowe.
# 3. Definicja cech
# Zmienna features definiuje listę kolumn, które będą używane jako cechy dla modelu.
# numeric_features i categorical_features są listami, które określają, które cechy są numeryczne, a które kategoryczne.
# 4. Przetwarzanie wstępne danych
# Pipeline: Definiuje ciąg operacji dla danych numerycznych (imputacja medianą i skalowanie) oraz kategorycznych (imputacja stałą wartością i one-hot encoding).
# ColumnTransformer: Umożliwia jednoczesne stosowanie różnych transformacji dla różnych kolumn danych wejściowych.
# 5. Przygotowanie danych
# fit_transform: Dopasowuje przekształcenia do danych i jednocześnie je transformuje, przygotowując macierz cech X oraz wektor etykiet y.
# 6. Podział danych na zbiory treningowe i walidacyjne
# train_test_split: Dzieli dane na zbiory treningowe i walidacyjne. Używa także indeksów z df_train dla zachowania odpowiednich identyfikatorów pasażerów.
# 7. Definicja modelu sieci neuronowej
# Sequential: Model sekwencyjny, gdzie warstwy są ułożone jedna po drugiej.
# Dense: Warstwy gęsto połączone, pierwsza z aktywacją ReLU, druga z sigmoid, które przewidują prawdopodobieństwo przynależności do klasy 1.
# 8. Kompilacja modelu
# Używa stochastycznego spadku gradientu (SGD) jako optymalizatora i entropii krzyżowej jako funkcji straty.
# 9. Trenowanie modelu
# Model jest trenowany na danych treningowych i jednocześnie walidowany na danych walidacyjnych.
# 10. Ewaluacja modelu
# Po zakończeniu treningu model jest oceniany na zbiorze walidacyjnym, gdzie obliczana jest dokładność klasyfikacji.
# 11. Predykcje
# Model dokonuje predykcji na zbiorze walidacyjnym, a wyniki są zapisywane w DataFrame, który jest następnie zapisywany do pliku CSV.
# 12. Wizualizacja
# Generowane są wykresy dokładności i funkcji strat dla zbiorów treningowych i walidacyjnych w trakcie treningu, co pozwala na monitorowanie postępu uczenia się modelu i dostosowanie hiperparametrów.