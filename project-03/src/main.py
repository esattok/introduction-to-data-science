import numpy as np
import matplotlib.pyplot as plt
from ann_hidden import ann_hidden
from Linear_Regressor import Linear_Regressor

# Calculate Standard Deviation
def calculate_std_dev(array, avg):
    return np.std(array, ddof=1)

# Read the file and store the values in a numpy array
def read_file(filename):
    data = np.loadtxt(filename, delimiter='\t')
    return data[:, 0], data[:, 1]

def hidden_select(hidden_units, train_input, train_output):
    print("Selecting the minimum hidden unit count")
    for i in hidden_units:
        iterations = 10000
        rate = 0.001
        ann = ann_hidden(i)
        ann.train(train_input, train_output, iterations, rate)
        print(f"For {i} hidden units:")
        ann.estimate(train_input, train_output)

def rate_of_learning(learning_rates, train_input, train_output):
    print("Selecting the best learning rate")
    for rate in learning_rates:
        iterations = 10000
        ann = ann_hidden(32)
        ann.train(train_input, train_output, iterations, rate)
        print(f"For {rate} learning rate:")
        ann.estimate(train_input, train_output)

def epoch_selection(epochs, train_input, train_output):
    print("Selecting the best epoch")
    for i in epochs:
        iterations = i
        rate = 0.001
        ann = ann_hidden(32)
        ann.train(train_input, train_output, iterations, rate)
        print(f"For {i} epochs:")
        ann.estimate(train_input, train_output)

def ann_evaluate(iterations, rate, hidden_units, train_input, train_output, test_input, test_output):
    ann = ann_hidden(hidden_units)
    ann.train(train_input, train_output, iterations, rate)
    ann.estimate(train_input, train_output)
    ann.plotting(train_input, train_output)
    ann.estimate(test_input, test_output)
    ann.plotting(test_input, test_output)

def ann_config(losses, train_input, train_output, test_input, test_output):
    iterations = 1000
    rate = 0.001
    ann = ann_hidden(2)
    ann.train(train_input, train_output, iterations, rate)
    print("The train loss for 2 units:")
    train_loss = ann.estimate(train_input, train_output)
    avg_loss = train_loss / len(train_input)
    losses[0, 0] = avg_loss
    losses[0, 1] = calculate_std_dev(train_input, avg_loss)
    ann.plotting(train_input, train_output)
    print("The test loss for 2 units:")
    test_loss = ann.estimate(test_input, test_output)
    avg_loss = test_loss / len(test_input)
    losses[1, 0] = avg_loss
    losses[1, 1] = calculate_std_dev(test_input, avg_loss)

    iterations = 10000
    rate = 0.0001
    ann = ann_hidden(4)
    ann.train(train_input, train_output, iterations, rate)
    print("The train loss for 4 units:")
    train_loss = ann.estimate(train_input, train_output)
    avg_loss = train_loss / len(train_input)
    losses[2, 0] = avg_loss
    losses[2, 1] = calculate_std_dev(train_input, avg_loss)
    ann.plotting(train_input, train_output)
    print("The test loss for 4 units:")
    test_loss = ann.estimate(test_input, test_output)
    avg_loss = test_loss / len(test_input)
    losses[3, 0] = avg_loss
    losses[3, 1] = calculate_std_dev(test_input, avg_loss)

    iterations = 100000
    rate = 0.001
    ann = ann_hidden(8)
    ann.train(train_input, train_output, iterations, rate)
    print("The train loss for 8 units:")
    train_loss = ann.estimate(train_input, train_output)
    avg_loss = train_loss / len(train_input)
    losses[4, 0] = avg_loss
    losses[4, 1] = calculate_std_dev(train_input, avg_loss)
    ann.plotting(train_input, train_output)
    print("The test loss for 8 units:")
    test_loss = ann.estimate(test_input, test_output)
    avg_loss = test_loss / len(test_input)
    losses[5, 0] = avg_loss
    losses[5, 1] = calculate_std_dev(test_input, avg_loss)

    iterations = 10000
    rate = 0.01
    ann = ann_hidden(16)
    ann.train(train_input, train_output, iterations, rate)
    print("The train loss for 16 units:")
    train_loss = ann.estimate(train_input, train_output)
    avg_loss = train_loss / len(train_input)
    losses[6, 0] = avg_loss
    losses[6, 1] = calculate_std_dev(train_input, avg_loss)
    ann.plotting(train_input, train_output)
    print("The test loss for 16 units:")
    test_loss = ann.estimate(test_input, test_output)
    avg_loss = test_loss / len(test_input)
    losses[7, 0] = avg_loss
    losses[7, 1] = calculate_std_dev(test_input, avg_loss)

    iterations = 10000
    rate = 0.001
    ann = ann_hidden(32)
    ann.train(train_input, train_output, iterations, rate)
    print("The train loss for 32 units:")
    train_loss = ann.estimate(train_input, train_output)
    avg_loss = train_loss / len(train_input)
    losses[8, 0] = avg_loss
    losses[8, 1] = calculate_std_dev(train_input, avg_loss)
    ann.plotting(train_input, train_output)
    print("The test loss for 32 units:")
    test_loss = ann.estimate(test_input, test_output)
    avg_loss = test_loss / len(test_input)
    losses[9, 0] = avg_loss
    losses[9, 1] = calculate_std_dev(test_input, avg_loss)

    # Print the losses
    print(losses)

train_input, train_output = read_file("train1.txt")
test_input, test_output = read_file("test1.txt")

# Linear Regression
reg = Linear_Regressor(train_input, train_output)
reg.regress_calc()
reg.line_plotting()

hidden_units = [2, 4, 8, 16, 32]
hidden_select(hidden_units, train_input, train_output)

learning_rates = [0.01, 0.001, 0.0001]
rate_of_learning(learning_rates, train_input, train_output)

epochs = [100, 1000, 10000, 100000, 1000000]
epoch_selection(epochs, train_input, train_output)

ann_evaluate(100000, 0.001, 32, train_input, train_output, test_input, test_output)

losses = np.zeros((10, 2))
ann_config(losses, train_input, train_output, test_input, test_output)
