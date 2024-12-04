import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    # ----------- above is given ------------------

    # bias = np.ones([n_data, 1])
    # train_data = np.c_[bias, train_data]
    train_data = np.hstack((np.ones((n_data, 1)), train_data))
    
    weights = initialWeights.reshape((n_features + 1, 1))
    z = np.dot(train_data, weights)
    y_pre = sigmoid(z)
    error = -np.sum(labeli * np.log(y_pre) + (1 - labeli) * np.log(1 - y_pre)) / n_data
    error_grad = np.dot(train_data.T, (y_pre - labeli)) / n_data

    error_grad = error_grad.flatten()  # transform to 1D array

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    n_data = data.shape[0]
    bias = np.ones([n_data, 1])
    data = np.c_[bias, data]
    # data: (N x (D+1)), W: ((D+1) x 10) -> probabilities: (N x 10)
    probabilities = sigmoid(np.dot(data, W))
    label = np.argmax(probabilities, axis=1).reshape(-1, 1)

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        Y: the label vector of size N x 10 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = 10


    bias = np.ones([n_data, 1])
    train_data = np.c_[bias, train_data]
    weights = params.reshape((n_feature + 1, n_class))
    z = np.dot(train_data, weights)
    y_pre = np.exp(z - np.max(z, axis=1, keepdims=True)) / np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1, keepdims=True) # softmax
    error = -np.sum(labeli*np.log(y_pre))/n_data
    error_grad = np.dot(train_data.T, (y_pre - labeli))/n_data
    error_grad = error_grad.flatten()


    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    n_data = data.shape[0]
    n_feature = data.shape[1]
    n_class = 10

    bias = np.ones([n_data, 1])
    data = np.c_[bias, data]
    weights = W.reshape((n_feature + 1, n_class))
    z = np.dot(data, weights)
    y_pre = np.exp(z - np.max(z, axis=1, keepdims=True)) / np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1,
                                                                  keepdims=True)  # softmax

    label = np.argmax(y_pre, axis=1).reshape(-1, 1)

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


if __name__ == "__main__":

    """
    Script for Logistic Regression
    """
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
    # number of classes
    n_class = 10

    # number of training samples
    n_train = train_data.shape[0]

    # number of features
    n_feature = train_data.shape[1]

    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()

    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros((n_feature + 1, 1)).flatten()
    # initialWeights = np.zeros((n_feature + 1,))
    opts = {'maxiter': 100}
    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        W[:, i] = nn_params.x.reshape((n_feature + 1,))

    # Find the accuracy on Training Dataset
    predicted_label = blrPredict(W, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    predicted_label = blrPredict(W, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    predicted_label = blrPredict(W, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    """
    # MY CODE
    # For Error Output of Problem 1
    training_error=[]
    testing_error=[]

    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        error, _ = blrObjFunction(W[:, i], *args)
        training_error.append(error)

        labeli_test = (test_label == i).astype(int).ravel().reshape(-1, 1)
        args_test = (test_data, labeli_test)
        error_test, _ = blrObjFunction(W[:, i], *args_test)
        testing_error.append(error_test)

    training_error = [float(e) for e in training_error]
    testing_error = [float(e) for e in testing_error]

    print("\nTraining Error for each class: ", training_error)
    print("\nTesting Error for each class: ", testing_error)

    classes = list(range(10)) 

    plt.figure(figsize=(10, 6))
    plt.bar(classes, training_error, width=0.4, label='Training Error', align='center', alpha=0.7)
    plt.bar([c + 0.4 for c in classes], testing_error, width=0.4, label='Testing Error', align='center', alpha=0.7)

    plt.xlabel('Class (Digit)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Training vs Testing Error for Each Class', fontsize=14)
    plt.xticks([c + 0.2 for c in n_class], classes)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.show()

    # ---------- MY CODE END HERE ----------------------------
    """

    """
    Script for Support Vector Machine
    """
    
    
    print('\n\n--------------SVM-------------------\n\n')
    ##################
    # YOUR CODE HERE #
    ##################


    sample = np.random.choice(len(train_data), 10000, replace=False)
    train_data_sample = train_data[sample]
    train_label_sample = train_label[sample]


    results = {}
    # linear kernel
    svc_linear = svm.SVC(kernel='linear')
    svc_linear.fit(train_data_sample, train_label_sample.ravel())
    train_acc = svc_linear.score(train_data_sample, train_label_sample)
    val_acc = svc_linear.score(validation_data, validation_label)
    test_acc = svc_linear.score(test_data, test_label)
    print("-----------------Linear Kenrel-----------------")
    print(f"training accuracy for linear kernel is {train_acc}")
    print(f"validation accuracy for linear kernel is {val_acc}")
    print(f"Testing accuracy for linear kernel is {test_acc}")
    # results['Linear Kernel'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}

    # radial basis with gamma=1
    svc_rbf1 = svm.SVC(kernel='rbf', gamma=1)
    svc_rbf1.fit(train_data_sample, train_label_sample.ravel())
    train_acc_rbf1 = svc_rbf1.score(train_data_sample, train_label_sample)
    val_acc_rbf1 = svc_rbf1.score(validation_data, validation_label)
    test_acc_rbf1 = svc_rbf1.score(test_data, test_label)
    print("------------------RBF Kernel with Gamma = 1-----------------")
    print(f"training accuracy for RBF Kernel with Gamma = 1 is {train_acc_rbf1}")
    print(f"validation accuracy for RBF Kernel with Gamma = 1 is {val_acc_rbf1}")
    print(f"Testing accuracy for RBF Kernel with Gamma = 1 is {test_acc_rbf1}")
    # results['RBF Kernel (gamma=1)'] = {'train': train_acc_rbf1, 'val': val_acc_rbf1, 'test': test_acc_rbf1}

    # radial basis with gamma set to default
    svc_rbf_default = svm.SVC(kernel='rbf')
    svc_rbf_default.fit(train_data_sample, train_label_sample.ravel())
    train_acc_rbf_default = svc_rbf_default.score(train_data_sample, train_label_sample)
    val_acc_rbf_default = svc_rbf_default.score(validation_data, validation_label)
    test_acc_rbf_default = svc_rbf_default.score(test_data, test_label)
    print("------------------RBF Kernel with Default Gamma-----------------")
    print(f"training accuracy for RBF Kernel with default Gamma is {train_acc_rbf_default}")
    print(f"validation accuracy for RBF Kernel with default Gamma is {val_acc_rbf_default}")
    print(f"Testing accuracy for RBF Kernel with default Gamma is {test_acc_rbf_default}")
    # results['RBF Kernel (gamma=default)'] = {'train': train_acc_rbf_default, 'val': val_acc_rbf_default, 'test': test_acc_rbf_default}

    # radial basis with default gamma but different C_value
    C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracy_results = []

    for C in C_values:
        svc_rbf = svm.SVC(kernel='rbf', C=C)
        svc_rbf.fit(train_data_sample, train_label_sample.ravel())
        train_acc_rbf = svc_rbf.score(train_data_sample, train_label_sample)
        val_acc_rbf = svc_rbf.score(validation_data, validation_label)
        test_acc_rbf = svc_rbf.score(test_data, test_label)
        accuracy_results.append({'C': C, 'train': train_acc_rbf, 'val': val_acc_rbf, 'test': test_acc_rbf})

    C_values = [result['C'] for result in accuracy_results]
    train_accs = [result['train'] for result in accuracy_results]
    val_accs = [result['val'] for result in accuracy_results]
    test_accs = [result['test'] for result in accuracy_results]

    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_accs, label='Training Accuracy', marker='o')
    plt.plot(C_values, val_accs, label='Validation Accuracy', marker='o')
    plt.plot(C_values, test_accs, label='Testing Accuracy', marker='o')
    plt.xlabel('C Value', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs C Value for RBF Kernel', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()

    print("# ----------- Optimal Solution ----------------------")
    # C =20, Default Gamma, RBF Kernel
    optimal_svc=svm.SVC(kernel='rbf', C=20)
    svc_rbf.fit(train_data, train_label.ravel())
    train_acc_optimal_svc = optimal_svc.score(train_data, train_label)
    val_acc_optimal_svc = optimal_svc.score(validation_data, validation_label)
    test_acc_optimal_svc = optimal_svc.score(test_data, test_label)
    print("------------------RBF Kernel with Default Gamma-----------------")
    print(f"training accuracy for Optimal SVC is {train_acc_optimal_svc}")
    print(f"validation accuracy for Optimal SVC is {val_acc_optimal_svc}")
    print(f"Testing accuracy for Optimal SVC is {test_acc_optimal_svc}")


    """
    Script for Extra Credit Part
    """
    print("-----------------MULTI LOGISTIC---------------------")
    # FOR EXTRA CREDIT ONLY
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1, n_class)).flatten()
    opts_b = {'maxiter': 100}

    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))

    # Find the accuracy on Training Dataset
    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    predicted_label_b = mlrPredict(W_b, validation_data)
    print(
        '\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
