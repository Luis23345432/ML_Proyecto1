import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
import h5py
import random

# Función para cargar datos de un archivo HDF5
def load_hdf5_dataset(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as hdf_file:
        if dataset_name:
            data = hdf_file[dataset_name][:]
            return data
        else:
            return {key: hdf_file[key][:] for key in hdf_file.keys()}

# Normalización de los datos
def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Implementación de KNN optimizado
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# Árbol de decisión (base para Random Forest)
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or depth >= self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        n_features = X.shape[1]
        best_split = None
        best_gain = -1

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)

        if best_split is None:
            return Counter(y).most_common(1)[0][0]

        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return {"feature": feature, "threshold": threshold, "left": left_child, "right": right_child}

    def _information_gain(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0

        p_left = len(y[left_mask]) / len(y)
        p_right = 1 - p_left
        gain = self._gini(y) - (p_left * self._gini(y[left_mask]) + p_right * self._gini(y[right_mask]))
        return gain

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature, threshold = tree["feature"], tree["threshold"]
        if x[feature] <= threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

# Random Forest optimizado
class RandomForest:
    def __init__(self, n_trees=50, max_depth=12, min_samples_split=5, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        for _ in tqdm(range(self.n_trees), desc="Entrenando árboles"):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            if self.max_features == "sqrt":
                n_selected_features = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_selected_features = int(np.log2(n_features))
            else:
                n_selected_features = n_features

            selected_features = np.random.choice(n_features, n_selected_features, replace=False)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, selected_features], y_sample)
            self.trees.append((tree, selected_features))

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)

# Cargar datos
train_data = load_hdf5_dataset('train.h5')
test_data = load_hdf5_dataset('test.h5')

X_train = np.hstack([train_data['body_acc_x'], train_data['body_acc_y'], train_data['body_acc_z'],
                     train_data['body_gyro_x'], train_data['body_gyro_y'], train_data['body_gyro_z'],
                     train_data['total_acc_x'], train_data['total_acc_y'], train_data['total_acc_z']])
y_train = train_data['y'].astype(int)
X_test = np.hstack([test_data['body_acc_x'], test_data['body_acc_y'], test_data['body_acc_z'],
                    test_data['body_gyro_x'], test_data['body_gyro_y'], test_data['body_gyro_z'],
                    test_data['total_acc_x'], test_data['total_acc_y'], test_data['total_acc_z']])

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Optimización de KNN
best_k = None
best_score = 0
for k in range(1, 11):
    knn = KNN(k=k)
    knn.fit(X_train_split, y_train_split)
    predictions = knn.predict(X_val)
    score = accuracy_score(y_val, predictions)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Mejor k para KNN: {best_k} con Accuracy: {best_score}")

# KNN final
knn = KNN(k=best_k)
knn.fit(X_train_split, y_train_split)
knn_predictions = knn.predict(X_val)

print("KNN Metrics:")
print("Accuracy:", accuracy_score(y_val, knn_predictions))
print("F1-Score:", f1_score(y_val, knn_predictions, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_val, knn_predictions))

# Optimización de Random Forest
best_params = None
best_rf_score = 0
for _ in range(10):  # Búsqueda aleatoria con 10 iteraciones
    params = {
        "n_trees": random.choice([50, 100, 150]),
        "max_depth": random.choice([10, 12, 15]),
        "min_samples_split": random.choice([2, 5, 10]),
    }
    rf = RandomForest(**params)
    rf.fit(X_train_split, y_train_split)
    predictions = rf.predict(X_val)
    score = accuracy_score(y_val, predictions)
    if score > best_rf_score:
        best_rf_score = score
        best_params = params

print(f"Mejores parámetros para Random Forest: {best_params} con Accuracy: {best_rf_score}")

# Random Forest final
rf = RandomForest(**best_params)
rf.fit(X_train_split, y_train_split)
rf_predictions = rf.predict(X_val)

print("Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_val, rf_predictions))
print("F1-Score:", f1_score(y_val, rf_predictions, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_val, rf_predictions))

# Predicciones finales
final_predictions = rf.predict(X_test)
submission = pd.DataFrame({
    'ID': np.arange(1, len(final_predictions) + 1),
    'Prediction': final_predictions
})
submission.to_csv('sample_submission.csv', index=False)
print("Archivo 'sample_submission.csv' generado correctamente.")
