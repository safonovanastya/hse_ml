class Tree:
    def leaf(data):
        return Tree(data=data)

    def __repr__(self):
        if self.is_leaf():
          return "Leaf(%r)" % self.data
        else:
          return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

    def __init__(self, *, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left == None and self.right == None

    def children(self):
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        return max([x.depth() for x in self.children()], default=0) + 1

import pandas as pd
import numpy as np

data = pd.read_csv('data.csv', sep=',', header=0, index_col = None)
data["ok"] = np.where(data["rating"] >= 0, True, False)

def single_feature_score(data, goal, feature):
    no = data[data[feature] == False]
    yes = data[data[feature] == True]
    if len(no) != 0 and len(no[no[goal] == True])/len(no) < 0.5:
        value_tp = False
        value_fp = True
    else:
        value_tp = True
        value_fp = False
    tp = no[no[goal] == value_tp]
    fp = yes[yes[goal] == value_fp]
    return (len(tp) + len(fp)) / (len(no) + len(yes))

def best_feature(data, goal, features):
  return max(features, key=lambda f: single_feature_score(data, goal ,f))

def worst_feature(data, goal, features):
  return min(features, key=lambda f: single_feature_score(data, goal ,f))

features = ['easy', 'ai', 'systems', 'theory', 'morning']
goal = 'ok'
print('The best feature is ' + best_feature(data, goal, features))
print('The worst feature is ' + worst_feature(data, goal, features))

def DecisionTreeTrain(data, features, maxdepth):
    for i in range(maxdepth):
        while len(features) != 0:
            best = best_feature(data, goal, features)
            false = data[data[best] == False]
            true = data[data[best] == True]
            features.remove(best)
            left = DecisionTreeTrain(false, features, maxdepth - 1)
            right = DecisionTreeTrain(true, features, maxdepth - 1)

            return Tree(data = data, left = left, right = right)

train = DecisionTreeTrain(data, features, 2)

def DecisionTreeTest(tree, test_point):
        if test_point == False:
            return DecisionTreeTest(left, test_point)
        if test_point == True:
            return DecisionTreeTest(right, test_point)

