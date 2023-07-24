import numpy as np
import matplotlib.pyplot as plt

class Linear_Regressor:
    def __init__(self, in_value, out_value):
        self.__input = in_value
        self.__output = out_value

    def loss_calc(self, predicted, real):
        return np.sum((predicted - real) ** 2)

    def regress_calc(self):
        x = np.vstack([self.__input, np.ones(len(self.__input))]).T
        self.__b_1, self.__b_0 = np.linalg.lstsq(x, self.__output, rcond=None)[0]

    def line_plotting(self):
        plt.style.use('fast')
        plt.style.use('Solarize_Light2')
        plt.style.use('grayscale')
        plt.scatter(self.__input, self.__output, label="Real", color='black', linestyle=":")
        predicted = self.__b_0 + self.__b_1 * self.__input
        loss = self.loss_calc(predicted, self.__output)
        plt.scatter(self.__input, predicted, label="Predicted", color='blue', linestyle="--")
        plt.xlabel('input')
        plt.ylabel('output')
        plt.legend()
        print("Loss ", loss)
        plt.title("Linear Regression: ")
        plt.show()
