import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

x_train = x_train - np.mean(x_train, axis = 0)
x_train = x_train / np.std(x_train,axis=0) + 1e-8

x_test = x_test - np.mean(x_test, axis = 0)
x_test = x_test / np.std(x_test,axis=0) + 1e-8

class LogisticRegressionModel:
    def __init__(self, learning_rate = 0.1, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def fit(self,x,y):
        n_samples,n_features = x.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(x,self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(x.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (_ + 1) % 100 == 0:
                print(f"weights{self.weights}",
                      f"bias{self.bias}")

    def predict(self,x):
        linear_model = np.dot(x,self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        y_classes = [
            1 if i > 0.5 else 0 for i in y_predicted
        ]

        return np.array(y_classes)

lr_model = LogisticRegressionModel(0.1,5000)    
lr_model.fit(x_train,y_train)

y_predicted_classes = lr_model.predict(x_test)

accuracy = accuracy_score(y_test,y_predicted_classes)

print("Accracy:", accuracy * 100, "%")