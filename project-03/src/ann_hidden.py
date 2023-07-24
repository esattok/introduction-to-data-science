import numpy as np
import random
import matplotlib.pyplot as plt

class ann_hidden:
    def __init__(self, node_cnt):
        self.__input_weight = np.random.random(node_cnt)
        self.__hidden_weight = np.random.random(node_cnt)
        self.__output_weight = 1 / float(node_cnt)

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def deriv(self, value):
        return value * (1 - value)

    def loss_calc(self, predicted, real):
        error_sum = 0
        for i in range( len(predicted) ):
            errorr = ( predicted[i] - real[i] ) ** 2
            error_sum += errorr
        return error_sum

    def __add_weight(self, layer, amount):
        if layer == "input":
            self.__input_weight += amount
        elif layer == "hidden":
            self.__hidden_weight += amount
        else:
            self.__output_weight += amount

    def train(self, in_value, out_value, step, rate):
        index = 0
        for _ in range(step):
            index %= len(in_value)

            i_w, h_w, o_w = self.__input_weight, self.__hidden_weight, self.__output_weight

            val = i_w + h_w * in_value[index]

            act = self.sigmoid(val)
            der_act = self.deriv(act)

            predicted = np.sum(o_w * act)

            err = out_value[index] - predicted
            in_amount = err * rate * o_w * der_act
            h_amount = in_amount * in_value[index]
            o_amount = err * rate * act

            self.__add_weight("input", in_amount)
            self.__add_weight("hidden", h_amount)
            self.__add_weight("output", o_amount)

            new_val = i_w + np.reshape(in_value, (len(in_value), 1)) * h_w
            act = self.sigmoid(new_val)
            predictions = np.dot(act, o_w)
            loss = self.loss_calc(predictions, out_value)

            index += 1

    def estimate(self, in_value, out_value):
        i_w, h_w, o_w = self.__input_weight, self.__hidden_weight, self.__output_weight

        val = i_w + np.reshape(in_value, (len(in_value), 1)) * h_w
        act = self.sigmoid(val)
        self.__predicted = np.dot(act, o_w)

        sum_loss = self.loss_calc(self.__predicted, out_value)
        print("The total loss is " + str(sum_loss))
        return sum_loss

    def plotting(self, in_value, out_value):
        sum_loss = self.loss_calc(self.__predicted, out_value)
        plt.style.use('fast')
        plt.style.use('Solarize_Light2')
        plt.style.use('grayscale')
        plt.scatter(in_value, self.__predicted, label="Predicted", color='blue', linestyle="--")
        plt.scatter(in_value, out_value, label="Real", color='green', linestyle="--")
        plt.xlabel("Input Values")
        plt.ylabel("Output Values")
        plt.legend()
        plt.title("Loss of ANN Data: " + str(sum_loss))
        plt.show()
