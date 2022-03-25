import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class MLP:
    def __init__(self, data, max_epoch, learning_rate):
        # obtención de datos
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.m, self.n = data.shape
        self.X = data[:, 0:self.n-1]
        self.Y = np.array([data[:, self.n-1]]).T

        # normalización de los datos
        self.mean = np.mean(self.X, axis=0)
        self.sigma = np.std(self.X, axis=0, ddof=1)
        self.X = stats.zscore(self.X, ddof=1)

        # numero de clases diferentes
        self.class_num = len(np.unique(self.Y))
        self.D = np.zeros((self.m, self.class_num))

        # guarda los errores por epoca
        self.error = []

        # matrices de pesos
        self.W1 = np.empty((10, self.n-1))
        self.W2 = np.empty((self.class_num, 10))
        self.init_weights()

    def init_weights(self):
        # matriz diagonal
        for i in range(self.m):
            self.D[i, self.Y[i]-1] = 1
        # pesos random en el rango de [-1 a 1]
        self.W1 = np.random.uniform(-1, 1, self.W1.shape)
        self.W2 = np.random.uniform(-1, 1, self.W2.shape)

    def sigmoid(self, y):
        return 1 / ( 1 + np.exp(-y))

    def softmax(self, y):
        return np.exp(y) / np.sum(np.exp(y))

    def feed_forward(self, x):        
        # capa oculta
        v1 = np.dot(self.W1, x);
        y1 = self.sigmoid(v1)

        # capa final
        v = np.dot(self.W2, y1)
        y = self.softmax(v)

        return y, y1

    def backpropagation(self, y, y1, d):
        # calculamos el error
        e = d - y
        delta = e
        self.error.append(np.sum(np.abs(e)))

        # retropropagamos el error
        e1 = np.dot(self.W2.T, delta)
        delta1 = y1 * (1 - y1) * e1

        return delta, delta1

    def train(self):
        conv = []
        for epoch in range(self.max_epoch):
            for i in range(self.m):
                x = np.array([self.X[i]]).T
                d = np.array([self.D[i]]).T

                # alimentamos la red
                y, y1 = self.feed_forward(x)

                # calculamos y retropropagamos el error
                delta, delta1 = self.backpropagation(y, y1, d)
                
                # actualizamos los pesos
                dW1 = self.learning_rate * np.dot(delta1, x.T)
                self.W1 = self.W1 + dW1

                dW2 = self.learning_rate * np.dot(delta, y1.T)
                self.W2 = self.W2 + dW2
            conv.append(np.sum(self.error[:]))
            print("Epoca: ", epoch)
            plt.plot(conv)
            self.error.clear() 
        plt.show()

    def encode_output(self, y):
        return (np.where(y == np.amax(y))[0][0]) + 1

    def evaluate(self, x, norm = True):
        if norm:
            x = (x - self.mean)/self.sigma;
        x = np.array([x]).T
        y,_ = self.feed_forward(x)
        return self.encode_output(y)

    def get_error_classification(self):
        errors = 0
        for i in range(self.m):
            y = self.evaluate(self.X[i], False)
            if self.Y[i, 0] == y:
                errors += 1
        return self.m - errors

if __name__ == "__main__":
    # obtenemos el set de datos
    df = pd.read_csv("dataset_multiclassOK.csv")
    # creamos una instancia de la red
    mlp = MLP(np.array(df), 200, 0.1)
    # entrenamos la red
    mlp.train()

    example_data = [47,2,3,6500,44300];
    print("Predicción: ", mlp.evaluate(example_data))
    print("Errores: ", mlp.get_error_classification())


    
