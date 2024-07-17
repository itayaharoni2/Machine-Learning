import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    # Extract the labels from the last column of the dataset
    labels = data[:, -1]

    # Find the unique classes and their counts, and puts them in arrays
    NameOfClassesArray, NumOfElements_ForEachClass = np.unique(labels, return_counts=True)

    # Calculate the total number of instances
    NumOfElements_InDataset = NumOfElements_ForEachClass.sum()

    # Calculate temporary Gini impurity
    for NumOfElements in NumOfElements_ForEachClass:
        p = NumOfElements / NumOfElements_InDataset
        gini += p * p
    
    gini = 1 - gini
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """

    entropy = 0.0
    # Extract the labels from the last column of the dataset
    labels = data[:, -1]

    # Find the unique classes and their counts, and puts them in arrays
    NameOfClassesArray, NumOfElements_ForEachClass = np.unique(labels, return_counts=True)

    # Calculate the total number of instances
    NumOfElements_InDataset = NumOfElements_ForEachClass.sum()

    # Calculate the Gini impurity
    for NumOfElements in NumOfElements_ForEachClass:
        p = NumOfElements / NumOfElements_InDataset
        if p > 0:  # to avoid log(0) which is undefined
            entropy += p * np.log2(p)  # Using log base 2
   
    # returns the negative of the sum
    return -entropy



def count_labels(data, feature):
    """Creates a dictionary of the labels and their count in a certain column in the data

    Args:
        data: the data to count the labels in
        feature: the index of the column

    Returns:
        dict: a dictionary of the labels and their count
    """
    Temp_array, counts = np.unique(data[:, feature], return_counts= True)
    return dict(zip(Temp_array,counts))






class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """

        # Assuming the last column of data contains the class labels
        labels = self.data[:, -1]
        # Counting the occurrences of each class label
        unique_classes, counts = np.unique(labels, return_counts=True)
        # Majority class is the one with the maximum count
        pred = unique_classes[np.argmax(counts)]
        return pred

        

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """

        self.children.append(node)
        self.children_values.append(val)


        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """

        goodness , _ = self.goodness_of_split(self.feature)
        self.feature_importance = (len(self.data) / n_total_sample) * goodness
    


    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.
        Optionally calculates the Gain Ratio if self.gain_ratio is True.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split or gain ratio
        - groups: a dictionary holding the data after splitting 
                according to the feature values.
        """
        # Extract the column corresponding to 'feature'
        feature_values = self.data[:, feature]
        # Dictionary to hold subsets of data for each unique feature value
        groups = {}
        # Populate the dictionary with subsets based on feature values
        for value in np.unique(feature_values):
            groups[value] = self.data[feature_values == value]

        # Initial impurity of the parent dataset
        initial_impurity = self.impurity_func(self.data)
        total_samples = len(self.data)
        weighted_impurity = 0
        split_info = 0  # Needed for Gain Ratio calculation

        # Calculate weighted impurity and split information
        for group_data in groups.values():
            proportion = len(group_data) / total_samples
            weighted_impurity += proportion * self.impurity_func(group_data)
            # Calculate part of the Split Information
            if proportion > 0 and self.gain_ratio:
                split_info -= proportion * np.log2(proportion)

        # Calculate goodness of the split
        goodness = initial_impurity - weighted_impurity

        # If gain_ratio is True, modify goodness to be the Gain Ratio
        if self.gain_ratio:
            # Avoid division by zero; handle the case where split_info is zero
            if split_info == 0:
                goodness = 0
            else:
                goodness /= split_info

        return goodness, groups



    def split(self):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to, and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        max_goodness = 0
        max_groups = []

        # Check if the current node has reached the maximum depth
        if self.depth >= self.max_depth:
            self.terminal = True
            return

        # Iterate over features to find the best one to split on
        for feature in range(self.data.shape[1] - 1):
            goodness_value, groups = self.goodness_of_split(feature)
            if goodness_value > max_goodness:
                max_groups = groups
                max_goodness = goodness_value
                self.feature = feature
        
        # If there's only one group or no improvement in splitting, mark node as terminal
        if len(max_groups) <= 1:
            self.terminal = True
            return
        
        # Create children nodes based on the best split
        for value, data in max_groups.items():
            new_node = DecisionNode(data, impurity_func=self.impurity_func, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(new_node, value)



    def calc_chi_value(self):
        """
        Calculate the chi value of the current node.

        Returns:
        - chi_value: the chi value of the node
        """
        
        chi_value = 0
        labels_dict = count_labels(self.data, -1)
        # calculate the probabilities of each label and store them in a dictionary
        probabilities = {label:num_of_labels / self.data.shape[0] for label,num_of_labels in labels_dict.items()}
        for child in self.children:
            child_labels_dict = count_labels(child.data, -1)
            Df = child.data.shape[0]
            for label in probabilities:
                E = Df * probabilities[label]
                if label in child_labels_dict:
                    num_of_instances = child_labels_dict[label]
                else:
                    num_of_instances = 0
                chi_value += (num_of_instances - E)**2 / E

        return chi_value




    def is_random(self):
        """
        Decides whether the current node split is random.

        Returns:
        - boolean: if the split was random
        """

        DOF = len(self.children_values) - 1 #degrees of freedom

        if self.chi == 1 or DOF < 1: 
            return False

        return self.calc_chi_value() < chi_table[DOF][self.chi]











class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        
        # Constructing the root and a queue which will hold all the nodes
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        queue = [self.root]

        # Continue until the queue is empty
        while queue:
            current_node = queue.pop(0)
            current_node.calc_feature_importance()
            # If no feature to split on, mark node as terminal
            if current_node.feature is None:
                current_node.terminal = True
                continue
            current_node.split()
            # If the node is random, mark it as terminal
            if current_node.is_random():
                current_node.terminal = True
                continue
            # Extend the queue with children nodes
            queue.extend(current_node.children)

        return self.root



    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None

        node = self.root
        while not node.terminal:
            feature_value = instance[node.feature]
            # Find child node for feature value
            if feature_value in node.children_values:
                node = node.children[node.children_values.index(feature_value)]
            else:
                break  # If value is not found, stop and use current node's prediction
        return node.pred




    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        correct_predictions = 0
        for instance in dataset:
            if self.predict(instance[:-1]) == instance[-1]:  # Compare predicted label to actual label
                correct_predictions += 1
        total_instances = len(dataset)
        accuracy = (correct_predictions / total_instances) * 100  # Convert to percentage
        return accuracy

 
    def depth(self):
        return self.root.depth()
















def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """

    training = []
    validation = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Assuming DecisionTree is a class that implements the decision tree logic
        # Create a new DecisionTree instance with the current max_depth
        tree = DecisionTree(X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        
        # Build the tree
        tree.build_tree()
        
        # Calculate accuracy on the training data
        train_accuracy = tree.calc_accuracy(X_train)
        training.append(train_accuracy)
        
        # Calculate accuracy on the validation data
        validation_accuracy = tree.calc_accuracy(X_validation)
        validation.append(validation_accuracy)
    

    return training, validation





def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        root = DecisionTree(X_train, impurity_func=calc_entropy, gain_ratio=True , chi=chi_val)
        root.build_tree()
        chi_training_acc.append(root.calc_accuracy(X_train))
        chi_testing_acc.append(root.calc_accuracy(X_test))
        depth.append(get_tree_depth(root.root))

    return chi_training_acc, chi_testing_acc, depth




def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    if node.terminal:
        return 1
    else:
        Number_of_nodes = 1
        for child in node.children:
            Number_of_nodes += count_nodes(child)
        return Number_of_nodes




def get_tree_depth(root):
    """Calculates the depth of a tree

    Args:
        root: the root of the tree

    Returns:
        Int: the depth of the tree
    """

    if root.terminal:
        return root.depth
    
    return max([get_tree_depth(child) for child in root.children])



