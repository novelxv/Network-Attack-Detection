import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2, p=3):
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
    def _compute_distance(self, x1, x2):
        if self.distance_metric == "euclidean":
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == "manhattan":
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == "minkowski":
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError("Invalid distance metric: {}".format(self.distance_metric))
    
    def predict(self, X):
        prediction = []
        for _, x in X.iterrows():
            distances = [self._compute_distance(x.values, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            prediction.append(most_common[0][0])
        return np.array(prediction)

    def get_params(self, deep=True):
        return {"k": self.k, "distance_metric": self.distance_metric}