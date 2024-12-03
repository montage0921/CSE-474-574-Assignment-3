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
    #----------- above is given ------------------

    bias = np.ones([n_data,1])
    train_data = np.c_[bias,train_data]
    weights = initialWeights.reshape((n_features + 1, 1))
    z = np.dot(train_data, weights)
    y_pre = sigmoid(z)
    error = -np.sum(labeli * np.log(y_pre) + (1 - labeli) * np.log(1 - y_pre)) / n_data
    error_grad = np.dot(train_data.T, (y_pre - labeli)) / n_data
    
    error_grad=error_grad.flatten() # transform to 1D array

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
    error = 0
    n_class = 10
    error_grad = np.zeros((n_feature + 1, n_class))


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
    label = np.zeros((data.shape[0], 1))

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
    
    # MY CODE
    training_error=[]
    testing_error=[]

    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    # initialWeights = np.zeros((n_feature + 1, 1))
    initialWeights = np.zeros((n_feature + 1, ))
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

    results={}
    # linear kernel
    svc_linear = svm.SVC(kernel='linear')
    svc_linear.fit(train_data, train_label.ravel())
    train_acc = np.mean(train_label == svc_linear.predict(train_data))
    val_acc = np.mean(validation_label == svc_linear.predict(validation_data))
    test_acc = np.mean(test_label == svc_linear.predict(test_data))
    results['Linear Kernel'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
    
    # radial basis with gamma=1
    svc_rbf1 = svm.SVC(kernel='rbf', gamma=1)
    svc_rbf1.fit(train_data, train_label.ravel())
    train_acc_rbf1 = np.mean(train_label == svc_rbf1.predict(train_data))
    val_acc_rbf1 = np.mean(validation_label == svc_rbf1.predict(validation_data))
    test_acc_rbf1 = np.mean(test_label == svc_rbf1.predict(test_data))
    results['RBF Kernel (gamma=1)'] = {'train': train_acc_rbf1, 'val': val_acc_rbf1, 'test': test_acc_rbf1}

    # radial basis with gamma set to default
    svc_rbf_default = svm.SVC(kernel='rbf')
    svc_rbf_default.fit(train_data, train_label.ravel())
    train_acc_rbf_default = np.mean(train_label == svc_rbf_default.predict(train_data))
    val_acc_rbf_default = np.mean(validation_label == svc_rbf_default.predict(validation_data))
    test_acc_rbf_default = np.mean(test_label == svc_rbf_default.predict(test_data))
    results['RBF Kernel (gamma=default)'] = {'train': train_acc_rbf_default, 'val': val_acc_rbf_default, 'test': test_acc_rbf_default}

    # radial basis with default gamma but different C_value
    C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracy_results = []

    for C in C_values:
        svc_rbf = svm.SVC(kernel='rbf', C=C)
        svc_rbf.fit(train_data, train_label.ravel())
        train_acc_rbf = np.mean(train_label == svc_rbf.predict(train_data))
        val_acc_rbf = np.mean(validation_label == svc_rbf.predict(validation_data))
        test_acc_rbf = np.mean(test_label == svc_rbf.predict(test_data))
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

    # print accuracy for first 3 svm models
    for key in results.keys():
        print(results[key])

    """
    Script for Extra Credit Part
    """
    # FOR EXTRA CREDIT ONLY
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    opts_b = {'maxiter': 100}
    
    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))
    
    # Find the accuracy on Training Dataset
    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
    
    # Find the accuracy on Validation Dataset
    predicted_label_b = mlrPredict(W_b, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
    
    # Find the accuracy on Testing Dataset
    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
    