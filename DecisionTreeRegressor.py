import numpy as np
import os
import json
import operator


def calculate_error(array):
    mean = np.mean(array)
    error = sum([(item - mean) ** 2 for item in array])
    return float(error)


def split(root, max_depth, min_samples_split, candidate_X, candidate_y, original_X):
    criteria_dict = dict()
    #print('candidate_X shape: {}'.format(candidate_X.shape))
    for j in range(original_X.shape[1]):
        for s in range(original_X.shape[0]):
            #print(j, s)
            #print(original_X[s, j])
            desired_position_small = np.where(candidate_X[:, j] <= original_X[s, j])
            desired_position_big = np.where(candidate_X[:, j] > original_X[s, j])
            #print('j: {}, s: {}, desired_position_big: {}'.format(j, s, desired_position_big))
            total_error = 0
            for item in [desired_position_big, desired_position_small]:
                if len(item[0]) == 0:
                    continue
                total_error += calculate_error(candidate_y[item])
            criteria_dict[(j, s)] = total_error
    sorted_dict = sorted(criteria_dict.items(), key=operator.itemgetter(1), reverse=False)
    #print('length of sorted dict is: {}'.format(len(sorted_dict)))
    #print(sorted_dict[: 10])
    #print(original_X[sorted_dict[0][0]])
    j = sorted_dict[0][0][0]
    s = sorted_dict[0][0][1]
    threshould = original_X[s, j]
    desired_position_small = np.where(candidate_X[:, j] <= threshould)
    desired_position_big = np.where(candidate_X[:, j] > threshould)
    #print('j: {}, s: {}, threshould: {}'.format(j, s, threshould))
    #print('desired_position_small: {}\n desired_position_big: {}'.format(desired_position_small, desired_position_big))

    #_______specify j and threshould_______
    root['splitting_variable'] = j
    root['splitting_threshold'] = threshould
    #print('left length: {}, right length: {}'.format(len(desired_position_small[0]), len(desired_position_big[0])))

    for index, sub_node in enumerate([desired_position_small, desired_position_big]):
        #_______ get current root_______
        current_candidate_X = None
        if index == 0:
            current_candidate_X = candidate_X[desired_position_small]
            current_candidate_y = candidate_y[desired_position_small]
        else:
            current_candidate_X = candidate_X[desired_position_big]
            current_candidate_y = candidate_y[desired_position_big]
        #print('current_candidate_X shape: {}'.format(current_candidate_X.shape))
        #_______ add node_______
        if len(sub_node[0]) < min_samples_split or root['depth'] == max_depth:
            if index == 0:
                root['left'] = np.mean(candidate_y[sub_node[0]])
            else:
                root['right'] = np.mean(candidate_y[sub_node[0]])
        else:
            if index == 0:
                root['left'] = dict()
                root['left']['depth'] = root['depth'] + 1
                split(root['left'], max_depth, min_samples_split, current_candidate_X, current_candidate_y, original_X)
            else:
                root['right'] = dict()
                root['right']['depth'] = root['depth'] + 1
                split(root['right'], max_depth, min_samples_split, current_candidate_X, current_candidate_y, original_X)


def explore(root, instance):
    j = root['splitting_variable']
    threshould = root['splitting_threshold']
    if instance[j] <= threshould:
        current_root = root['left']
    else:
        current_root = root['right']
    if isinstance(current_root, dict) is False:  # means it is a leaf node
        return current_root
    else:  # means it is a internal node
        return explore(current_root, instance)


class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        self.root = dict()
        self.root['depth'] = 1
        print('min_samples_split is: {}'.format(self.min_samples_split))
        split(self.root, self.max_depth, self.min_samples_split, X, y, X)

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        answer = []
        for i in range(X.shape[0]):
            instance = X[i, :]
            pred = explore(self.root, instance)
            answer.append(pred)
        return np.array(answer)

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test
if __name__ == '__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) + ".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) + ".csv", delimiter=",")

        for j in range(2):
            print('---i: {}, j: {}---'.format(i, j))
            #print(np.where(x_train[:, 7] <= 0.39177 &
            #               x_train[:, 7] > 0.049539 &
            #               x_train[:, 10] <= 0.188457 &
            #               x_train[:, 19] <= 0.479295 &
            #               x_train[:, 0] <= 0.632774))
            #x_train = x_train[x_train[:, 7] <= 0.39177]
            #x_train = x_train[x_train[:, 7] > 0.049539]
            #x_train = x_train[x_train[:, 10] <= 0.188457]
            #x_train = x_train[x_train[:, 19] <= 0.479295]
            #x_train = x_train[x_train[:, 0] <= 0.632774]
            #print('Filtered x_train: {}'.format(x_train))
            #if i != 1 or j != 0: continue
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_string = tree.get_model_string()
            #print('model_string is\n{}'.format(model_string))

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            #print('test model_string is\n{}'.format(test_model_string))
            #print('length model: {}, length test: {}'.format(len(str(model_string)), len(str(test_model_string))))
            #print(str(model_string) == str(test_model_string))
            #print(operator.eq(model_string, test_model_string))
            #print('Model same? {}'.format(operator.eq(model_string, test_model_string)))

            #sometimes although the some node is different, but the structure of leaf node is the same!
            y_pred = tree.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_" + str(i) + "_" + str(j) + ".csv", delimiter=",")
            #print('y_test_pred is:{}'.format(y_test_pred))
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
            print('Error is: {}'.format(np.square(y_pred - y_test_pred).mean()))
