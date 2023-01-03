# import necessary lib
from math import log
import numpy as np


class Decision_Tree_C45:
    # define entropy calculation
    def Entropy(self, train_data):
        inst_num = len(train_data)  # instances number
        label_counts = {}  # count instances of each class
        for i in range(inst_num):

            label = train_data[i][-1]  # get instance class
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1  # count
        ent = 0
        for key in label_counts.keys():
            # calculate each class proportion
            prob = float(label_counts[key]) / inst_num
            ent -= prob * log(prob, 2)  # see Eq.(3.1)
        return ent
        # split data according to feature and feature value

    def Gini(self, train_data):
        inst_num = len(train_data)  # instances number
        label_counts = {}  # count instances of each class
        for i in range(inst_num):
            label = train_data[i][-1]  # get instance class
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1  # count
        prob_mul = 0
        for key in label_counts.keys():
            prob = float(label_counts[key]) / inst_num
            prob_mul += prob * prob
        return 1 - prob_mul

    def split_data(self, train_data, feature_index, feature_value, feature_type):
        splitedData = []
        if feature_type == "D":
            for feat_vect in train_data:
                if feat_vect[feature_index] == feature_value:
                    reducedVect = []
                    for i in range(len(feat_vect)):
                        if i < feature_index or i > feature_index:
                            reducedVect.append(feat_vect[i])
                    splitedData.append(reducedVect)
        if feature_type == "L":
            for feat_vect in train_data:
                if feat_vect[feature_index] <= feature_value:
                    splitedData.append(feat_vect)
        if feature_type == "R":
            for feat_vect in train_data:
                if feat_vect[feature_index] > feature_value:
                    splitedData.append(feat_vect)
        return splitedData

    def choose_split_feature_gini(self, train_data):
        feat_num = len(train_data[0]) - 1
        bestGini = np.inf
        best_feat_index = -1
        best_feat_value = 0
        for i in range(feat_num):
            if isinstance(train_data[0][i], str):  # for discrete feature
                feat_list = [example[i] for example in train_data]
                unique_values = set(feat_list)
                newGini = 0

                for value in unique_values:
                    sub_data = self.split_data(train_data, i, value, "D")
                    prop = float(len(sub_data)) / len(train_data)
                    newGini += prop * self.Gini(sub_data)

                if newGini < bestGini:
                    best_feat_index = i
                    bestGini = newGini

            else:  # for continous feature
                feat_list = [example[i] for example in train_data]
                unique_values = set(feat_list)
                sort_unique_values = sorted(unique_values)

                for j in range(len(sort_unique_values) - 1):
                    div_value = (sort_unique_values[j] + sort_unique_values[j + 1]) / 2
                    sub_data_left = self.split_data(train_data, i, div_value, "L")
                    sub_data_right = self.split_data(train_data, i, div_value, "R")
                    prop_left = float(len(sub_data_left)) / len(train_data)
                    prop_right = float(len(sub_data_right)) / len(train_data)
                    newGini = prop_left * self.Gini(sub_data_left) + \
                              prop_right * self.Gini(sub_data_right)
                    if newGini < bestGini:
                        bestGini = newGini
                        best_feat_index = i
                        best_feat_value = div_value

        return best_feat_index, best_feat_value

    def choose_split_feature(self, train_data):
        feat_num = len(train_data[0]) - 1
        base_ent = self.Entropy(train_data)
        bestInforGain = 0.0
        best_feat_index = -1
        best_feat_value = 0
        for i in range(feat_num):
            if isinstance(train_data[0][i], str):  # for discrete feature
                feat_list = [example[i] for example in train_data]
                unique_values = set(feat_list)
                newEnt = 0

                for value in unique_values:
                    sub_data = self.split_data(train_data, i, value, "D")
                    prop = float(len(sub_data)) / len(train_data)
                    newEnt += prop * self.Entropy(sub_data)  # see Eq.(3.2)

                inforgain = base_ent - newEnt
                if inforgain > bestInforGain:
                    best_feat_index = i
                    bestInforGain = inforgain

            else:  # for continous feature
                feat_list = [example[i] for example in train_data]
                unique_values = set(feat_list)
                sort_unique_values = sorted(unique_values)

                for j in range(len(sort_unique_values) - 1):
                    div_value = (sort_unique_values[j] + sort_unique_values[j + 1]) / 2
                    sub_data_left = self.split_data(train_data, i, div_value, "L")
                    sub_data_right = self.split_data(train_data, i, div_value, "R")
                    prop_left = float(len(sub_data_left)) / len(train_data)
                    prop_right = float(len(sub_data_right)) / len(train_data)
                    ent = prop_left * self.Entropy(sub_data_left) + \
                          prop_right * self.Entropy(sub_data_right)
                    inforgain = base_ent - ent
                    if inforgain > bestInforGain:
                        bestInforGain = inforgain
                        best_feat_index = i
                        best_feat_value = div_value

        return best_feat_index, best_feat_value

    def create_decision_tree(self, train_data, feat_names, depth=-1, minimal_data=-1):
        classList = [example[-1] for example in train_data]
        if classList.count(classList[0]) == len(classList):  # see condition A
            return classList[0]
        if len(train_data[0]) == 1:  # see condition B
            return max(set(classList), key=classList.count)
        if len(train_data) == 0:  # see condition C
            return
        if depth == 0:
            return max(set(classList), key=classList.count)
        if minimal_data > 0 and len(train_data) < minimal_data:
            return max(set(classList), key=classList.count)
        best_feat, best_div_value = self.choose_split_feature_gini(train_data)
        if isinstance(train_data[0][best_feat], str):  # for discrete feature
            feat_name = feat_names[best_feat]
            tree_model = {feat_name: {}}  # generate a root node
            del (feat_names[best_feat])  # del feature used
            feat_values = [example[best_feat] for example in train_data]
            unique_feat_values = set(feat_values)
            # create a node for each value of the best feature
            for value in unique_feat_values:
                sub_feat_names = feat_names[:]
                tree_model[feat_name][value] = \
                    self.create_decision_tree(self.split_data(train_data,
                                                              best_feat, value, "D"),
                                              sub_feat_names, depth - 1, minimal_data)
        else:
            best_feat_name = feat_names[best_feat] + "<" + str(best_div_value)
            tree_model = {best_feat_name: {}}  # generate a root node
            sub_feat_names = feat_names
            new_data_left = self.split_data(train_data,
                                            best_feat,
                                            best_div_value,
                                            "L")
            new_data_right = self.split_data(train_data,
                                             best_feat,
                                             best_div_value, "R")
            tree_model[best_feat_name]["Y"] = \
                self.create_decision_tree(new_data_left, sub_feat_names, depth - 1, minimal_data)
            tree_model[best_feat_name]["N"] = \
                self.create_decision_tree(new_data_right, sub_feat_names, depth - 1, minimal_data)
        return tree_model

    def predict(self, tree_model, feat_names, feat_vect):
        firstStr = list(tree_model.keys())[0]
        lessIndex = str(firstStr).find('<')
        if lessIndex > -1:
            secondDict = tree_model[firstStr]
            feat_name = str(firstStr)[:lessIndex]
            featIndex = feat_names.index(feat_name)
            div_value = float(str(firstStr)[lessIndex + 1:])
            if feat_vect[featIndex] <= div_value:
                if isinstance(secondDict["Y"], dict):
                    classLabel = self.predict(secondDict["Y"],
                                              feat_names, feat_vect)
                else:
                    classLabel = secondDict["Y"]
            else:
                if isinstance(secondDict["N"], dict):
                    classLabel = self.predict(secondDict["N"],
                                              feat_names, feat_vect)
                else:
                    classLabel = secondDict["N"]
            return classLabel
        else:
            secondDict = tree_model[firstStr]
            featIndex = feat_names.index(firstStr)
            key = feat_vect[featIndex]
            valueOfFeat = secondDict[key]
            if isinstance(valueOfFeat, dict):
                classLabel = self.predict(valueOfFeat, feat_names, feat_vect)
            else:
                classLabel = valueOfFeat
            return classLabel
