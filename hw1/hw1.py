###### Your ID ######
# ID1: 208277574
# ID2: 208082735
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    
    X_mean = np.mean(X , axis = 0)
    X_minMax = np.max(X , axis = 0) - np.min(X , axis = 0)
    X_norm = (X - X_mean) / X_minMax

    y_mean = np.mean(y , axis = 0)
    y_minMax = np.max(y , axis = 0) - np.min(y , axis = 0)
    y_norm = (y - y_mean) / y_minMax

    return X_norm, y_norm


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    
    # Create a column vector of ones (m x 1)
    ones = np.ones(X.shape[0])
    
    # Concatenate the column vector of ones to the original matrix X
    # np.hstack stacks arrays in sequence horizontally (column wise).
    X_with_bias = np.column_stack((ones, X))
    
    return X_with_bias


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features). Assume X includes the bias if necessary.
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model including bias as the first term.

    Returns:
    - J: The cost associated with the current set of parameters (single number).
    """
    J = 0
    # Number of training examples
    m = X.shape[0]
    
    # Predicted values
    y_expected = X.dot(theta)
    
    # Squared errors
    sq_errors = (y_expected - y) ** 2
    
    # Cost function J
    J = 1 / (2 * m) * np.sum(sq_errors)
    
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    
    for _ in range(num_iters):
        J_history.append(compute_cost(X, y, theta))
        vec = (X.dot(theta) - y)
        gradient = vec.dot(X) / X.shape[0]
        theta = theta - gradient * alpha
    
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    # Step 1: Compute the transpose of X
    X_transpose = np.transpose(X)
    
    # Step 2: Compute the product of X transpose and X
    XTX = X_transpose.dot(X)
    
    # Step 3: Compute the inverse of XTX
    XTX_inverse = np.linalg.inv(XTX)
    
    # Step 4: Compute the product of XTX inverse and X transpose
    pinv_theta = XTX_inverse.dot(X_transpose).dot(y)

    return pinv_theta



def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    prev_cost = float('inf')  # Initialize the previous cost with infinity
    
    for _ in range(num_iters):
        # Compute the cost function value for the current parameters
        cost = compute_cost(X, y, theta)
        J_history.append(cost)  # Append the current cost to the history
        
        # Check if the improvement in the cost function is smaller than the threshold
        if prev_cost - cost < 1e-8:
            break  # If so, stop the learning process
        
        # Update the previous cost for the next iteration
        prev_cost = cost
        
        # Compute the gradient of the cost function
        gradient = (X.dot(theta) - y).dot(X) / len(y)
        
        # Update the parameters (weights) using the gradient descent update rule
        theta -= alpha * gradient
    
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    
    np.random.seed(42)
    random_theta = np.random.random(X_train.shape[1])

    for alpha in alphas:
        theta, _ = efficient_gradient_descent(X_train, y_train, random_theta, alpha, iterations)# train the model using the selected alpha
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)# compute the loss on the validation set 
    return alpha_dict



def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    features_list = [i for i in range(X_train.shape[1])] # list of the indexes valid features to be selected
    for i in range(5):
        np.random.seed(42)
        theta_guess = np.random.random(i+2) # creates a random theta vector
        costs_dict = {}
        for feature in features_list:
            selected_features.append(feature) # add the feature to the selected features
            X_candidate = apply_bias_trick(X_train[:, selected_features])
            theta, _ = efficient_gradient_descent(X_candidate, y_train, theta_guess, best_alpha, iterations)# train the model using the selected feature
            costs_dict[feature] = compute_cost(apply_bias_trick(X_val[:, selected_features]), y_val, theta)# compute the loss on the validation set for the selected feature
            selected_features.remove(feature)# remove the feature from the selected features
        
        min_feature = min(costs_dict, key = costs_dict.get) # get the feature with the minimum loss of validation 
        selected_features.append(min_feature) # add the feature to the selected features
        features_list.remove(min_feature)# remove the feature from the list of valid features to be selected

    return selected_features



def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    # Make a copy of the input dataframe to avoid modifying the original
    df_poly = df.copy()
    
    # Iterate over each column in the dataframe
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i:]:
            # Create new features by squaring each feature
            if col1 == col2:
                new_col = df[col1] ** 2
                new_col.name = f"{col1}^2"  # Set the name of the new feature
                df_poly = pd.concat([df_poly, new_col], axis=1)  # Concatenate the new feature to the dataframe
            # Create new features by taking the product of pairs of features
            else:
                new_col = df[col1] * df[col2]
                new_col.name = f"{col1}*{col2}"  # Set the name of the new feature
                df_poly = pd.concat([df_poly, new_col], axis=1)  # Concatenate the new feature to the dataframe
    
    return df_poly