import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import h5py
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

# Función para cargar datos de un archivo HDF5
def load_hdf5_dataset(file_path):
    with h5py.File(file_path, 'r') as hdf_file:
        return {key: hdf_file[key][:] for key in hdf_file.keys()}

# Normalización de los datos
def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Implementación de KNN optimizado con KD-Tree
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.tree = None
        self.y_train = None

    def fit(self, X, y):
        self.tree = KDTree(X)
        self.y_train = y

    def predict(self, X):
        distances, indices = self.tree.query(X, k=self.k)
        if self.k == 1:
            indices = np.expand_dims(indices, axis=-1)
        predictions = []
        for neighbors in indices:
            k_labels = self.y_train[neighbors]
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

# Random Forest optimizado con paralelización
class RandomForest:
    def __init__(self, n_trees=150, max_depth=20, min_samples_split=4, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _fit_single_tree(self, X, y):
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        if self.max_features == "sqrt":
            n_selected_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_selected_features = int(np.log2(n_features))
        else:
            n_selected_features = n_features

        selected_features = np.random.choice(n_features, n_selected_features, replace=False)
        tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(X_sample[:, selected_features], y_sample)
        return tree, selected_features

    def fit(self, X, y):
        self.trees = Parallel(n_jobs=-1)(
            delayed(self._fit_single_tree)(X, y) for _ in tqdm(range(self.n_trees), desc="Entrenando árboles")
        )

    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X[:, features]) for tree, features in self.trees
        ])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)

# Cargar datos
train_data = load_hdf5_dataset('train.h5')
test_data = load_hdf5_dataset('test.h5')

# Preprocesamiento de datos
X_train = np.hstack([
    train_data['body_acc_x'], train_data['body_acc_y'], train_data['body_acc_z'],
    train_data['body_gyro_x'], train_data['body_gyro_y'], train_data['body_gyro_z'],
    train_data['total_acc_x'], train_data['total_acc_y'], train_data['total_acc_z']
])
y_train = train_data['y'].astype(int)
X_test = np.hstack([
    test_data['body_acc_x'], test_data['body_acc_y'], test_data['body_acc_z'],
    test_data['body_gyro_x'], test_data['body_gyro_y'], test_data['body_gyro_z'],
    test_data['total_acc_x'], test_data['total_acc_y'], test_data['total_acc_z']
])

# Normalización
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Reducción de dimensionalidad
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Dividir el conjunto de entrenamiento para validación
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Entrenamiento con los mejores hiperparámetros para Random Forest
rf = RandomForest(n_trees=365, max_depth=None, min_samples_split=2)
rf.fit(X_train_split, y_train_split)
rf_val_predictions = rf.predict(X_val)

# Métricas para Random Forest
rf_accuracy = accuracy_score(y_val, rf_val_predictions)
rf_f1 = f1_score(y_val, rf_val_predictions, average='weighted')
rf_confusion_matrix = confusion_matrix(y_val, rf_val_predictions)

print("Random Forest Metrics:")
print(f"Accuracy: {rf_accuracy}")
print(f"F1-Score: {rf_f1}")
print("Confusion Matrix:\n", rf_confusion_matrix)

# Predicción final para Random Forest
rf_predictions = rf.predict(X_test)
rf_submission = pd.DataFrame({
    'ID': np.arange(1, len(rf_predictions) + 1),
    'Prediction': rf_predictions
})
rf_submission.to_csv('random_forest_submission.csv', index=False)

# Entrenamiento y predicción para KNN
knn = KNN(k=1)
knn.fit(X_train_split, y_train_split)
knn_val_predictions = knn.predict(X_val)

# Métricas para KNN
knn_accuracy = accuracy_score(y_val, knn_val_predictions)
knn_f1 = f1_score(y_val, knn_val_predictions, average='weighted')
knn_confusion_matrix = confusion_matrix(y_val, knn_val_predictions)

print("KNN Metrics:")
print(f"Accuracy: {knn_accuracy}")
print(f"F1-Score: {knn_f1}")
print("Confusion Matrix:\n", knn_confusion_matrix)

# Predicción final para KNN
knn_predictions = knn.predict(X_test)
knn_submission = pd.DataFrame({
    'ID': np.arange(1, len(knn_predictions) + 1),
    'Prediction': knn_predictions
})
knn_submission.to_csv('knn_submission.csv', index=False)

print("Archivos 'random_forest_submission.csv' y 'knn_submission.csv' generados correctamente.")
