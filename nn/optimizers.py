from utils.data_partitioners import mini_batch_partitioner
from sklearn.model_selection import train_test_split

class SGD():

    def __init__(self, network, partitioner=mini_batch_partitioner, learning_rate=0.3, batch_size=32, n_iterations=30):
        self.network = network

        self.learning_rate = learning_rate
        self.partitioner = partitioner
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iterations = n_iterations

    def __call__(self, X, y, test_size=0.2):
        X_train, X_validation, T_train, T_validation = train_test_split(X, y, test_size=test_size)
        training_set = (X_train, T_train)
        validation_set = (X_validation, T_validation)

        XT_batches, X_validation, T_validation = self.partitioner(training_set, validation_set, self.batch_size)
        validation_costs = []

        for i in xrange(self.n_iterations):
            print("Started iteration: {} of {}".format(i+1, self.n_iterations))
            for X, Y in XT_batches:
                Y_predicted = self.network.feedforward(X)
                self.network.backpropagate(self.learning_rate, Y_predicted, Y)

            Y_predicted = self.network.feedforward(X_validation)
            validation_cost = self.network.cost(T_validation, Y_predicted)
            print(validation_cost)
            validation_costs.append(validation_cost)

            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    print("Cost did not decrease after three iterations! Quitting.")
                    return validation_costs

        return validation_costs

