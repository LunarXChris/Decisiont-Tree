
import math
import random
import csv
import sys

class Random_forest:
    def __init__(self, dataset, depth=5, num_of_tree=10, bootstrap=50, num_of_feature=10, algo='gini'):
        self.dataset = dataset
        self.depth = depth
        self.num_of_tree = num_of_tree
        if bootstrap < len(self.dataset):
            self.bootstrap = bootstrap
        else:
            self.bootstrap = len(self.dataset)
        if num_of_feature < len(self.dataset[0]) - 1:
            self.num_of_feature = num_of_feature
        else:
            self.num_of_feature = len(self.dataset[0]) - 1
        self.algo = algo
        self.train_samples = None
        self.forest = []

    # draw bootstrap samples of size n with replacement
    def forest_sampling(self):
        train_samples = list()
        dataset_copy = list(self.dataset)
        for i in range(self.num_of_tree):
            random.shuffle(dataset_copy)
            train_samples.append(dataset_copy[:self.bootstrap])
        return train_samples

    # train n decision trees with limit num of features
    def train_forest(self):
        self.train_samples = self.forest_sampling()
        for subset in range(self.num_of_tree):
            d_tree = Decision_tree(self.train_samples[subset], self.depth, self.num_of_feature, self.algo)
            d_tree.build_tree()
            self.forest.append(d_tree)

        for x in self.forest:
            print("---------------------------")
            x.print_tree(x.tree)

    # evaluate the accuracy of random forest algorithm
    def eval_forest(self, dataset):
        predicted = []

        for sample in dataset:
            is_zero, is_one = 0, 0
            for d_tree in self.forest:
                prediction = d_tree.predict(d_tree.tree, sample)
                if prediction == 0:
                    is_zero += 1
                else:
                    is_one += 1
            if is_zero >= is_one:
                predicted.append(0)
            else:
                predicted.append(1)

        actual = [row[-1] for row in dataset]
        accuracy = self.accuracy_metric(actual, predicted)
        print("accuracy = " + str(accuracy))

    # calculate the accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


class Decision_tree:
    def __init__(self,  dataset, depth, num_of_feature=5, algo='gini'):
        self.dataset = dataset
        self.depth = depth
        self.num_of_feature = num_of_feature
        self.algo = algo
        self.min_size = 10
        self.tree = None
        self.used_indices = []

    # make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # print a decision tree
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * 2 * ' ', (node['index']), node['value'])))
            if node["index"] <= -1:
                print('%s' % ((depth * 2 * ' ')) + str(node))
            else:
                if node.get('left'):
                    print('%s_left' % ((depth * 2 * ' ')))
                    self.print_tree(node['left'], depth + 1)
                if node.get('right'):
                    print('%s_right' % ((depth * 2 * ' ')))
                    self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * 2 * ' ', node)))

    # build a decision tree
    def build_tree(self):
        root = self.get_split_groups(self.dataset)
        self.tree = root
        self.node_split(root, 1)
        return self.tree

    # return the majority of leaf node
    def leaf_node(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # split the tree
    def node_split(self, node, depth):
        # print('depth: ' + str(depth))
        # self.print_tree(self.tree)
        # print(node['groups'])

        # out of indices
        if node['index'] == -2:
            node['left'] = node['right'] = self.leaf_node(node['groups'])
            del (node['groups'])
            return

        left, right = node['groups']
        del(node['groups'])

        # all same class
        if node['score'] == 0.0:
            node['left'] = node['right'] = self.leaf_node(left + right)
            return

        # check for max depth
        if depth >= self.depth:
            node['left'], node['right'] = self.leaf_node(left), self.leaf_node(right)
            return

        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.get_split_groups(left)
            self.node_split(node['left'], depth+1)

        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.leaf_node(right)
        else:
            node['right'] = self.get_split_groups(right)
            self.node_split(node['right'], depth+1)
    
    # return best split groups with evaluation function
    def get_split_groups(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        index_, value_, score_, groups_ = 999, 999, -1, None

        if len(self.used_indices) == len(dataset[0]) - 1:
            return {'index': -2, 'value': -2, 'score': -1, 'groups': dataset}

        if self.algo == 'entropy':
            base_score = self.entropy([dataset], class_values)
            print('Base Entropy=%.3f' % (base_score))
        else:
            base_score = self.gini([dataset], class_values)
            print('Base Gini=%.3f' % (base_score))

        # randomly choose limited num of feature of test split without replacement
        indices = [x for x in range(len(dataset[0]) - 1)]
        random.shuffle(indices)
        num_of_iter = 0
        for index in indices:
            if num_of_iter >= self.num_of_feature:
                break
            elif index in self.used_indices:
                continue
            num_of_iter += 1

            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                if self.algo == 'entropy':
                    score = self.entropy(groups, class_values)
                    print('X%d < %.3f Entropy=%.3f' % (index, row[index], score))
                else:
                    score= self.gini(groups, class_values)
                    print('X%d < %.3f Gini=%.3f' % (index, row[index], score))
                if score == 0.0:
                    print("same_class")
                    return {'index': -1, 'value': -1, 'score': score, 'groups': groups}
                info_gain = base_score - score
                print("info_gain: " + str(info_gain))
                if info_gain > score_:
                    index_, value_, score_, groups_ = index, row[index], info_gain, groups
        print("score_: " + str(score_))
        self.used_indices.append(index_)
        if groups_ is None:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {'index': index_, 'value': value_, 'score': score_, 'groups': groups_}

    # split the dataset with a testing feature
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # calculate gini for a group of nodes
    def gini(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 1.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                if p > 0:
                    score -= p * p
            # weight the group score by its relative size
            gini += score * (size / n_instances)
        return gini

    # calculate entropy for a group of nodes
    def entropy(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        entropy = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                if p > 0:
                    score -= p * math.log(p)
            entropy += score * (size / n_instances)
        return entropy


car_mapping = {
        'buying': {'low': 2, 'med': 3, 'high': 4, 'vhigh': 5},
        'maint': {'low': 2, 'med': 3, 'high': 4, 'vhigh': 5},
        'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
        'persons': {'2': 2, '4': 4, 'more': 5},
        'lug_boot': {'small': 2, 'med': 3, 'big': 4},
        'safety': {'low': 2, 'high': 3, 'med': 4},
        'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    }

# load, combine and change data format of dataset X and Y
def load_csv(file_X, file_Y):
    with open(file_X, 'r') as train_attr:
        lines = csv.reader(train_attr)
        dataset_x = list(lines)
    if "car" in file_X:
        for row in range(1, len(dataset_x)):
            for col in range(len(dataset_x[0])):
                label = dataset_x[0][col]
                val = dataset_x[row][col]
                dataset_x[row][col] = car_mapping[label][val]
    # print(dataset_x)

    with open(file_Y, 'r') as label:
        lines = csv.reader(label)
        dataset_y = list(lines)
    if "car" in file_X:
        for row in range(1, len(dataset_y)):
            for col in range(len(dataset_y[0])):
                label = dataset_y[0][col]
                val = dataset_y[row][col]
                dataset_y[row][col] = car_mapping[label][val]
    # print(dataset_y)

    dataset = []
    for i in range(1, len(dataset_x)):
        dataset.append(dataset_x[i] + dataset_y[i])

    # convert string attributes to float
    if "car" not in file_X:
        for column in range(len(dataset[0])):
            for row in dataset:
                row[column] = float(row[column].strip())
    return dataset


def main():
    if len(sys.argv) < 3:
        return
    # load and prepare data
    train_x = sys.argv[1]
    train_y = sys.argv[2]

    dataset = load_csv(train_x, train_y)

    if len(sys.argv) == 8: # gini or entropy
        random_forest = Random_forest(dataset, int(sys.argv[3]), int(sys.argv[4]),int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
    elif len(sys.argv) == 7: # num_of_features
        random_forest = Random_forest(dataset, int(sys.argv[3]), int(sys.argv[4]),int(sys.argv[5]), int(sys.argv[6]))
    elif len(sys.argv) == 6: # bootstrap_size
        random_forest = Random_forest(dataset, int(sys.argv[3]), int(sys.argv[4]),int(sys.argv[5]))
    elif len(sys.argv) == 5: # num_of_tree
        random_forest = Random_forest(dataset, int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) == 4: # max_depth
        random_forest = Random_forest(dataset, int(sys.argv[3]))
    else:
        random_forest = Random_forest(dataset)

    random_forest.train_forest()
    random_forest.eval_forest(random_forest.dataset)

    test_x = input("\ninput_path_of_test_dataset_x: ")
    print("Confirm: ", test_x)
    test_y = input("\ninput_path_of_test_dataset_y: ")
    print("Confirm: ", test_y)
    test_dataset = load_csv(test_x, test_y)

    random_forest.eval_forest(test_dataset)


if __name__ == '__main__':
    main()