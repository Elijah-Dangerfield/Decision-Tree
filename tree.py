
import numpy as np

class Node():
    def __init__(self, value = None, feature = None, terminal = False):
        self.feature = feature #feature col
        self.value = value # value at which to split at
        self.terminal = terminal
        self.left = None
        self.right = None
        self.categorization = ""
    def __str__(self):
        result ="\n".join(["Feature: "+ str(self.feature),"Value: " + str(self.value),"Category: "+ str(self.categorization), "Terminal: " + str(self.terminal)])
        if self.left is not None and not self.left.terminal:
            result = result + "\nLeft: Exists"
        else:
            left = "\nLeft :Terminal"
            result = result + left

        if self.right is not None and not self.right.terminal:
            result = result + "\nRight: Exists"
        else:
            right = "\nRight :Terminal"
            result = result + right
        return result

def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    sum = np.sum(counts)
    return np.sum([(-counts[i]/sum)*np.log2(counts[i]/sum) for i in range(len(elements))])


def find_split(data, features, target):
    split = None
    best_gain = 0
    for x in features:
        (max_gain, best_split) = split_for_max_gain(data, x, target)
        if max_gain > best_gain:
            best_gain = max_gain
            split = (x, best_split)

    return Node(split[1], split[0])


def split_for_max_gain(data, feature, target_name):
    sorted = data.sort_values(feature)
    values = np.unique(sorted[feature])
    splits = [(values[x] + values[x+1])/2 for x in range(len(values)-1)]
    max_gain = 0
    split = 0
    for x in splits:
        gain = info_gain_split(data, x, feature,target_name)
        if gain > max_gain:
            max_gain = gain
            split = x
    return (max_gain,split)


def info_gain_split(data, split_value, split_feature, target_feature):
    total_entropy = entropy(data[target_feature])

    above = []
    below = []

    for x in data[split_feature]:
        if x >= split_value:
            above.append(x)
        else:
            below.append(x)

    total = len(above) + len(below)

    entropy_above = (len(above) / total) * entropy(
        data.where(data[split_feature] > split_value).dropna()[target_feature])
    entropy_below = (len(below) / total) * entropy(
        data.where(data[split_feature] < split_value).dropna()[target_feature])

    weighted_entropy = entropy_above + entropy_below
    return total_entropy - weighted_entropy

def no_possible_splits(data, features):
    for feature in features:
        if(len(np.unique(data[feature])) > 1):
            return False
    return True