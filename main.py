import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def init2(S, K1, K2):
    W1 = np.random.uniform(-0.1, 0.1, (K1, S + 1))  # Dodanie 1 dla biasu
    W2 = np.random.uniform(-0.1, 0.1, (K2, K1 + 1))  # Dodanie 1 dla biasu
    return W1, W2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dzialaj2(W1, W2, X):
    # X powinien już mieć bias dodany przed wywołaniem
    U1 = np.dot(W1, X)
    Y1 = sigmoid(U1)
    Y1_bias = np.insert(Y1, 0, 1)  # Dodanie biasu do wyników z pierwszej warstwy
    U2 = np.dot(W2, Y1_bias)
    Y2 = sigmoid(U2)
    return Y1, Y2  # Zwróć Y1 bez dodawania dodatkowego biasu

def ucz2(W1, W2, P, T, n, alpha=0.1, beta=0.9):
    mse_history = []
    for i in range(n):
        mse_epoch = []
        for j in range(P.shape[1]):
            X = np.insert(P[:, j], 0, 1)  # Dodajemy bias
            t = T[j]
            Y1, Y2 = dzialaj2(W1, W2, X)
            E2 = t - Y2
            D2 = E2 * Y2 * (1 - Y2)
            E1 = np.dot(W2[:, 1:].T, D2)
            D1 = E1 * Y1 * (1 - Y1)
            delta_W1 = alpha * np.outer(D1, X)
            delta_W2 = alpha * np.outer(D2, np.insert(Y1, 0, 1))
            W1 += delta_W1
            W2 += delta_W2
            mse_epoch.append(np.mean(np.square(E2)))
        mse_history.append(np.mean(mse_epoch))
    return W1, W2, mse_history

def predict(W1, W2, X):
    X_with_bias = np.insert(X, 0, 1)  # Dodanie biasu na początku
    _, Y2 = dzialaj2(W1, W2, X_with_bias)
    return (Y2 > 0.5).astype(int)


# Wczytanie danych
df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')

# Zachowanie dodatkowych informacji
passenger_info = df_train[['PassengerId', 'Name', 'Age', 'Sex', 'Pclass']]

# Analiza statystyczna
print(df_train.describe())
print(df_train['Pclass'].value_counts())

# Przetwarzanie danych
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Przetwarzanie danych treningowych
X = preprocessor.fit_transform(df_train[features])
y = df_train['Survived']

# Podział danych na zbiór treningowy i walidacyjny z zachowaniem indeksów dla PassengerId
X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X, y, passenger_info, test_size=0.2, random_state=42
)

# Inicjalizacja sieci neuronowej
input_size = X_train.shape[1]
hidden_size = 5
output_size = 1
W1, W2 = init2(input_size, hidden_size, output_size)

# Uczenie sieci
W1, W2, mse_history = ucz2(W1, W2, X_train.T, y_train.values, n=1000, alpha=0.01, beta=0.9)

# Wykres błędu MSE
plt.plot(mse_history)
plt.title('Błąd MSE na przestrzeni epok')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.show()

# Ewaluacja modelu
y_pred = np.array([predict(W1, W2, X_val[i, :]) for i in range(X_val.shape[0])])
accuracy = accuracy_score(y_val, y_pred)
print(f"Dokładność modelu: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, cmap='Blues')
plt.title('Macierz błędów')
plt.xlabel('Przewidywane etykiety')
plt.ylabel('Prawdziwe etykiety')
plt.show()

# Zakładamy, że y_pred zawiera już predykcje modelu, a df_train to oryginalny DataFrame

# Wyciągamy PassengerId dla zbioru walidacyjnego
#passenger_ids = df_train.iloc[X_val.index]['PassengerId']

# Przygotowanie DataFrame z predykcjami
additional_info = df_train.loc[X_val.index, ['Name', 'Age', 'Sex', 'Pclass']]
predictions = pd.DataFrame({
    'PassengerId': df_train.loc[X_val.index, 'PassengerId'],
    'Name': additional_info['Name'],
    'Age': additional_info['Age'],
    'Sex': additional_info['Sex'],
    'Pclass': additional_info['Pclass'],
    'Survived': y_pred
})

# Zapisywanie przewidywań do pliku CSV
predictions.to_csv('survivors_predictions.csv', index=False)
print("Przewidywane etykiety przeżycia zostały zapisane do pliku 'survivors_predictions.csv'.")