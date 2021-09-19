import numpy as np


class Perceptron():
    def __init__(self, lr, epochs):
        self.weights = np.random.randn(3)*1e-4
        print("Before training: ", self.weights)
        self.lr = lr
        self.epochs = epochs

    def activation(self, inputs):
        z = np.dot(inputs, self.weights)
        return np.where(z > 0, 1, 0)

    def fit(self, x, y):
        self.x = x
        self.y = y

        x_with_bias = np.c_[self.x, -np.ones((len(self.x), 1))]
        print(f"x with bias: {x_with_bias}")

        for epoch in range(self.epochs):
            print('--'*10)
            print(f"for epoch: {epoch}")
            print('--'*10)

            y_hat = self.activation(x_with_bias)

            self.error = self.y - y_hat

            self.weights = self.weights + self.lr * \
                np.dot(x_with_bias.T, self.error)
            print("error: ", self.error)
            print(f"Updated weights: {self.weights}")
            print("####"*10)

    def predict(self, X):
        x_with_bias = np.c_[X, np.ones((len(X), 1))]
        return self.activation(x_with_bias)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total loss: {total_loss}")
        return total_loss
