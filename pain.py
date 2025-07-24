import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

# split dataset into train and test
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape # m_train is the number of training examples

def init_params():
    hidden_layer_1_size = 128 # Increased from 10
    hidden_layer_2_size = 64  # Increased from 10
    output_size = 10 # Number of output classes (0-9)
    input_size = X_train.shape[0] # Number of features (784 pixels)

    W1 = np.random.randn(hidden_layer_1_size, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((hidden_layer_1_size, 1))

    W2 = np.random.randn(hidden_layer_2_size, hidden_layer_1_size) * np.sqrt(2. / hidden_layer_1_size)
    b2 = np.zeros((hidden_layer_2_size, 1))

    W3 = np.random.randn(output_size, hidden_layer_2_size) * np.sqrt(2. / hidden_layer_2_size)
    b3 = np.zeros((output_size, 1))

    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0) # Allows for more complex situations

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X): #first results
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z): #relu algorithm
    return Z > 0

def one_hot(Y): #changes label into an array
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y): #fixes the errors using past results to fix
    m_batch = Y.size # Use the number of examples in the current batch/full set
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m_batch * dZ3.dot(A2.T)
    db3 = 1 / m_batch * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m_batch * dZ2.dot(A1.T)
    db2 = 1 / m_batch * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_batch * dZ1.dot(X.T)
    db1 = 1 / m_batch * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha): # adds the derivatives from backward prop 
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3): #find predicrtions
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y) # Commented out as it spams output
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations): #everything together now
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 100 == 0: #PRINT ACCURACY EVERY 100
            print("Iteration: ", i)
            predictions_train = get_predictions(A3)
            print(f"Training Accuracy: {get_accuracy(predictions_train, Y)*100:.2f}%")
            _, _, _, _, _, A3_dev = forward_prop(W1, b1, W2, b2, W3, b3, X_dev)
            predictions_dev = get_predictions(A3_dev)
            print(f"Dev Accuracy: {get_accuracy(predictions_dev, Y_dev)*100:.2f}%")
            print("-" * 30)
    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, alpha=0.15, iterations=1) # Increased iterations, slight alpha adjustment

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions
#predictions

#find actual results
def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print(f"\n--- Test Prediction for Index {index} ---")
    print("Prediction: ", prediction[0])
    print("Actual Label: ", label)
    current_image_reshaped = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image_reshaped, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, Actual: {label}")
    plt.show()

# Test some predictions (using the functions from previous answer)
print("\n--- Testing Individual Predictions ---")
test_prediction(0, W1, b1, W2, b2, W3, b3)
test_prediction(1, W1, b1, W2, b2, W3, b3)
test_prediction(2, W1, b1, W2, b2, W3, b3)
test_prediction(3, W1, b1, W2, b2, W3, b3)
test_prediction(100, W1, b1, W2, b2, W3, b3)