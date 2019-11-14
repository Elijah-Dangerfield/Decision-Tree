from tree import *
import pandas as pd

def ID3(data, originaldata, features, target):
    if (no_possible_splits(data, features)):
        node = Node(terminal=True)
        vals, counts = np.unique(data[target], return_counts=True)
        category_index = 0
        for i in range(len(vals)):
            if (counts[i] > counts[category_index]):
                category_index = i
        node.categorization = vals[category_index]
        return node

    if len(np.unique(data[target])) == 1:
        node = Node(terminal=True)
        node.categorization = np.unique(data[target])[0]
        return node

    root = find_split(data, features, target)

    # grabs all rows where split feautre is below
    left_data = data.where(data[root.feature] < root.value).dropna()

    # grabs all rows where split feautre is above
    right_data = data.where(data[root.feature] > root.value).dropna()

    root.left = ID3(left_data, originaldata, features, target)
    root.right = ID3(right_data, originaldata, features, target)

    return root


def main():
    data = pd.read_csv('/Users/elidangerfield/Documents/school/CSCI4350/OLA3/iris-data.txt', sep=" ",
                       header=None)
    data.columns = ["sepal_length", "sepal_width", "pedal_length", "pedal_width", "class"]
    features = ["sepal_length", "sepal_width", "pedal_length", "pedal_width"]
    target = "class"
    node = ID3(data, data, features, target)
    print(node.right.right.left.right)



if __name__ == '__main__':
    main()