import numpy as np
import gzip
import pickle
import time
import ffnn as ff


def main():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        data = pickle._Unpickler(f)
        data.encoding = 'latin1'
        train, valid, test = data.load()

    x_train, y_train2 = train
    x_valid, y_valid = valid
    x_test, y_test = test

    # Transform Y to an array of len 10
    y_train = np.zeros((len(y_train2), 10))
    for i in range(0, len(y_train2)):
        y_train[i] = [0]*10
        a = y_train2[i]
        y_train[i][a] = 1

    # Training settings
    epochs = 5
    batch = 50
    learnrate = 5

    # Initilizing the neural network
    network = ff.ffnn(784+1, 30+1, 10)

    # Training the neural network
    startTime = time.time()
    network.train(x_train, y_train, x_valid, y_valid,
                  epochs, batch, learnrate)

    # Testing the nural network with unsean data
    type, result = network.validate(x_test, y_test)
    print("\nResult of final network\nAccuracy = ",
          round(result*100, 2), "\nTotal runtime: ", time.time()-startTime, "seconds")


if __name__ == "__main__":
    main()
