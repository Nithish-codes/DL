import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

class NeuralNetwork:
    def __init__(self, learning_rate = 0.1, num_iterations = 1000, hidden_size = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.hidden_size = hidden_size

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh_derivative(self, a):
        return 1 - a ** 2

    def fit(self, x,y):
        n_samples, n_features = x.shape

        self.w1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.b1 = np.zeros((1,self.hidden_size))            

        self.w2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = 0

        y = y.reshape(-1,1)

        for i in range(self.num_iterations):
            z1 = np.dot(x,self.w1) + self.b1
            a1 = np.tanh(z1)

            z2 = np.dot(a1,self.w2) + self.b2
            a2 = self.sigmoid(z2)

            dz2 = a2 - y

            dw2 = np.dot(a2.T, dz2) / n_samples
            db2 = np.sum(dz2) / n_samples

            dz1 = np.dot(dz2, self.w2.T)
            dz1 = dz1 * self.tanh_derivative(a1)

            dw1 = np.dot(x.T, dz1) / n_samples
            db1 = np.sum(dz1) / n_samples

            self.w1 -= self.learning_rate * dw1
            self.b1 -= self.learning_rate * db1
            self.w2 -= self.learning_rate * dw2
            self.b2 -= self.learning_rate * db2

    def predict(self, x):            
        z1 = np.dot(x,self.w1) + self.b1
        a1 = np.tanh(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        y_predicted = self.sigmoid(z2)

        y_predicted_class = [
            1 if i > 0.5 else 0 for i in y_predicted
        ]

        return np.array(y_predicted_class).reshape(-1)

model = NeuralNetwork(0.1, 5000, 10)
model.fit(x_train,y_train)

y_predicted = model.predict(x_test)

accuracy = accuracy_score(y_predicted, y_test)

print(f"Accuracy:{accuracy * 100:.2f}%")
            