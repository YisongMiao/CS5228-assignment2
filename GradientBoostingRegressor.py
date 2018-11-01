import numpy as np
from DecisionTreeRegressor import MyDecisionTreeRegressor, explore
import os
import json
import operator


def loss_function(a, b):
    assert len(a) == len(b)
    sum = 0
    for i in range(len(a)):
        sum += 0.5 * (a[i] - b[i]) ** 2
    return sum


class MyGradientBoostingRegressor():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param learning_rate: type:float
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        int (default=100)
        :param n_estimators: type: integer
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        :param max_depth: type: integer
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node

        estimators: the regression estimators
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.start_point = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.estimators in this function
        '''
        tree = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(X, y)
        f_prev = np.empty(y.shape)
        f_prev.fill(np.mean(y))
        self.start_point = np.mean(y)

        #print('y_pred is: {}'.format(y_pred))
        #print('y_true is: {}'.format(y))
        for i in range(self.n_estimators):
            residual = y - f_prev
            #print('residual is: {}'.format(residual))
            residual_tree = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            residual_tree.fit(X, residual)
            residual_tree.explore_track(X)
            #print('track_dict is: {}'.format(residual_tree.track_dict))
            self.estimators[i] = residual_tree
            for k, v in residual_tree.track_dict.items():
                y_true = y[v['index']]
                y_pred = f_prev[v['index']]
                c = np.mean(y_true) - np.mean(y_pred)
                c *= self.learning_rate
                #new = y_pred + c
                #loss = 0.5 * sum([item ** 2 for item in (y_true - new)])
                #loss = loss_function(y_true, new)
                #v['cm'] = self.learning_rate * loss
                v['cm'] = c

                #print('k: {}, v:{}'.format(k, v))
                #print('y_true: {}'.format(y_true))
                #print('y_pred: {}'.format(y_pred))
                #print('c: {}'.format(c))
                #print('new: {}'.format(new))
                #print('loss: {}'.format(loss))

            #_______ update f_prev_______
            additive = np.zeros(y.shape)
            for k, v in residual_tree.track_dict.items():
                for index in v['index']:
                    additive[index] = v['cm']
            #print(f_prev)
            f_prev += additive
            #print(f_prev)

            #print('additive: {}'.format(additive))
            #print('f_prev: {}'.format(f_prev))

    def predict_instance(self, instance):
        y_pred = self.start_point
        for i in range(len(self.estimators)):
            track = []
            explore(self.estimators[i].root, instance, track)
            pattern = ''.join(track)
            cm = self.estimators[i].track_dict[pattern]['cm']
            #print('cm is: {}'.format(cm))
            y_pred += cm
        return y_pred

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        answer = []
        for i in range(X.shape[0]):
            instance = X[i, :]
            pred = self.predict_instance(instance)
            answer.append(pred)
        return np.array(answer)

    def get_model_string(self):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i): self.estimators[i].root})
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i): self.estimators[i].root})

        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test
if __name__ == '__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) + ".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) + ".csv", delimiter=",")

        for j in range(2):
            # ^^^!!! Be cautious and need to remove it!!!%%%
            #if i != 1 or j != 0: continue
            # ^^^!!! Be cautious and need to remove it!!!%%%
            print('i is: {}, j is: {}'.format(i, j))
            n_estimators = 10 + j * 10
            gbr = MyGradientBoostingRegressor(n_estimators=n_estimators, max_depth=5, min_samples_split=2)
            gbr.fit(x_train, y_train)
            model_string = gbr.get_model_string()

            with open("Test_data" + os.sep + "gradient_boosting_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            print(operator.eq(model_string, test_model_string))
            for k in range(len(model_string)):
                print('#{} tree: {}'.format(k, operator.eq(model_string[str(k)], test_model_string[str(k)])))

            y_pred = gbr.predict(x_train)

            gbr.save_model_to_json('test.json')

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_gradient_boosting_" + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
