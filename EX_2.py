import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0) + 1e-8

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

class LogisticRegressionModel:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        
        self.weights = np.zeros(n_features)
        self.bias = 0

        
        for iteration in range(self.num_iterations):

            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            cost = -np.mean(
                y * np.log(y_predicted + 1e-8)
                + (1 - y) * np.log(1 - y_predicted + 1e-8)
            )

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.num_iterations}"
                    f" | Cost: {cost:.4f}"
                )

    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(linear_model)

        predictions = [
            1 if probability > 0.5 else 0
            for probability in probabilities
        ]

        return np.array(predictions)


lr_model = LogisticRegressionModel(
    learning_rate=0.001,
    num_iterations=5000
)

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 40)
print(f"Model Accuracy on Test Set : {accuracy:.4f}")
print("=" * 40)
