import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math as math

neuralNetworkArch = [2, 5, 5, 5, 5, 1]


def load_data():
    fileX1 = open("trainX1DrawingBook.txt", 'r')
    fileX2 = open("trainX2DrawingBook.txt", 'r')
    fileY = open("trainYDrawingBook.txt", 'r')

    X1 = list(fileX1.read().split("\n"))
    X2 = list(fileX2.read().split("\n"))
    Y = list(fileY.read().split("\n"))

    X1 = list(map(int, X1[0:X1.__len__() - 1]))
    X2 = list(map(int, X2[0:X2.__len__() - 1]))
    Y = list(map(int, Y[0:Y.__len__() - 1]))

    X = np.array([X1[0:int(X1.__len__() * (90 / 100))], X2[0:int(X2.__len__() * (90 / 100))]])
    Xtest = np.array([X1[0:int(X1.__len__() * (10 / 100))], X2[0:int(X2.__len__() * (10 / 100))]])

    Y = np.array(Y[0:int(Y.__len__() * (90 / 100))])
    Ytest = np.array(Y[0:Xtest.shape[1]])

    Y = Y.reshape((1, Y.shape[0]))
    Ytest = Ytest.reshape((1, Ytest.shape[0]))

    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)

    return X, Y, Xtest, Ytest


def placeHolders(x, y):
    X = tf.placeholder(tf.float32, shape=(x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(y, None), name="Y")

    return X, Y


def initializeParameters(dims):
    parameters = {}
    for i in range(1, dims.__len__()):
        parameters["W" + str(i)] = tf.get_variable("W" + str(i), (dims[i], dims[i - 1]),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters["b" + str(i)] = tf.get_variable("b" + str(i), (dims[i], 1), initializer=tf.zeros_initializer())

    return parameters


def forwardProp(X, parameters, dims):
    Z = []
    A = []
    Z.append(tf.add(tf.matmul(parameters["W1"], X), parameters["b1"]))
    A.append(tf.nn.relu(Z[0]))
    for i in range(2, dims.__len__()):
        Z.append(tf.add(tf.matmul(parameters["W" + str(i)], A[i - 2]), parameters["b" + str(i)]))
        A.append(tf.nn.relu(Z[i - 1]))

    return Z[dims.__len__() - 2]


def compCost(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.abs(tf.reduce_sum(Y - ZL))

    return cost


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]
    mini_batches = []

    # print(m)
    permutation = list(np.random.permutation(m))
    # print(permutation)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(1, m)

    num_complete_minibatches = math.floor(
        m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.00001, num_epochs=1200, minibatch_size=64, print_cost=True):
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = placeHolders(n_x, n_y)

    parameters = initializeParameters(neuralNetworkArch)

    Z3 = forwardProp(X, parameters, neuralNetworkArch)

    cost = compCost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        dictionary = {'hello': 'world'}
        np.save('my_file.npy', parameters)

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        # x1 = float(input("In 1"))
        # x2 = float(input("In 2"))
        # y = ((forwardProp([[x1], [x2]], parameters, [2, 5, 5, 5, 5, 1])))
        # tf.Print(y, [y], message="ans: ")

        return parameters


def restore():
    tf.reset_default_graph()

    read_dictionary = np.load('my_file.npy').item()
    # tf.convert_to_tensor(read_dictionary,np.float32)
    """"
    v1 = tf.get_variable("v1", shape=[3])
    with tf.Session() as sess:
        print("Model restored.")
        saver.restore(sess, "C:\\SSD\\PythonPrograms\\TrainData\\DrawingBook\\model.ckpt")
        parameters = sess.run('par:0')
    """
    x1 = float(input("Input 1: "))
    x2 = float(input("Input 2: "))
    y = ((forwardProp([[x1], [x2]], read_dictionary, neuralNetworkArch)))
    with tf.Session() as session:
        print("Ans: ", np.round(session.run(y)))


ch = int(input("\nWould you link to train the network or test the last trained model?(Enter 1 or 2)\n"))
if ch==1:
    X, Y, Xtest, Ytest = load_data()
    model(X, Y, Xtest, Ytest)
else:
    restore()
