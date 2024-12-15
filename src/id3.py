import numpy as np
import pandas as pd

class Node():
    def __init__(self, name: str, isResult: bool):
        self.name : str = name
        self.children : dict[str, Node] = {}
        self.isResult : bool = isResult
    
    def print(self, ind: int):
        print(" "*ind + self.name)
        for c in self.children:
            print(" "*(ind) + "-" + str(c))
            self.children[c].print(ind+2)

class ID3():
    def __init__(self, random_seed: int = 0):
        self.start : Node = Node('', False)
        self.colVal: dict[str, list[str]] = {}
        self.thresholds : dict[str, float] = {} 
        self.random_seed = random_seed
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        x_train = X.copy()
        for c in x_train.columns:
            if (x_train[c].dtype != 'object') and (x_train[c].nunique() > 2):
                split = self.__discretize(x_train[c], y)
                self.thresholds.update({c: split})
                x_train[c] = x_train[c] >= split
            
            self.colVal.update({c: x_train[c].unique()})
        self.start = self.__decision_tree_learning(x_train, y, None)
    
    def predict(self, X: pd.DataFrame):
        xc = X.copy()
        for c in self.thresholds:
            xc[c] = xc[c] >= self.thresholds[c]
        result = []
        for row in xc.iterrows():
            node = self.start
            while not node.isResult:
                node = node.children.get(row[1][node.name])
            result.append(node.name)
        return result

    def __entropy(self, s: pd.Series) -> float:
        vcount = s.value_counts(normalize=True, sort=False)
        return -(vcount*np.log2(vcount)).sum()
    
    def __information_gain(self, x: pd.Series, y: pd.Series) -> float:
        vcount = x.value_counts(normalize=True, sort=False)
        ent = [self.__entropy(y[x[x == val].index]) for val in vcount.index]
        return (vcount*ent).sum()

    def __plurality_value(self, y: pd.Series):
        modes = y.mode()
        return modes[self.random_seed % len(modes)]
    
    def __best_attr(self, x: pd.DataFrame, y: pd.Series) -> str:
        return x.columns[np.argmin([self.__information_gain(x[col], y) for col in x.columns])]
    
    def __discretize(self, x: pd.Series, y: pd.Series):
        d = pd.concat([x, y], axis=1)
        d = d.sort_values(by=[x.name], ascending=True)
        avg_of_y = [d[d[y.name] == val][x.name].mean() for val in y.unique()]
        avg_between_y = [(avg_of_y[i]+avg_of_y[i+1])/2 for i in range (len(avg_of_y)-1)]

        best_gain = 9999
        best_split = 0
        for val in avg_between_y:
            gain = self.__information_gain(x >= val, y)
            if gain < best_gain:
                best_gain = gain
                best_split = val
        
        return best_split

    def __decision_tree_learning(self, x: pd.DataFrame, y: pd.Series, yParent: pd.Series | None):
        if len(y) == 0:
            return Node(self.__plurality_value(yParent), True)
        elif (y.nunique() == 1):
            return Node(y.head(1).values[0], True)
        elif x.empty:
            return Node(self.__plurality_value(y), True)
        else:
            col = self.__best_attr(x, y)
            tree = Node(col, False)
            for val in self.colVal[col]:
                exs = x[x[col] == val].drop(columns=[col])
                subtree = self.__decision_tree_learning(exs, y[exs.index], y)
                tree.children[val] = subtree
            return tree
