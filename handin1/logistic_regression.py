import numpy as np
from h1_util import numerical_grad_check
import math

def logistic(z):
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """
    logi = np.zeros(z.shape)
    ### YOUR CODE HERE 1-5 lines
    logi= 1/(1+np.exp(-z))
    ### END CODE
    assert logi.shape == z.shape
    return logi


class LogisticRegressionClassifier():

    def __init__(self):
        self.w = None

    def cost_grad(self, X, y, w):
        """
        Compute the average cross entropy and the gradient under the logistic regression model 
        using data X, targets y, weight vector w 
        
        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features 
           y: np.array shape (n,)  int - Labels 
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar the cross entropy cost of logistic regression with data X,y 
           grad: np.arrray shape(n,d) gradient of cost at w 
        """
        cost = 0
        grad = np.zeros(w.shape)
        temp = 0
        self.w = w
        ### YOUR CODE HERE 5 - 15 lines
       # m = len(X)
       # for i in range (m):
       #     x= y*np.log(logistic(np.dot(np.transpose(w),X[i]))) + (1-y)*np.log(1-logistic(np.dot(np.transpose(w),X[i])) )
       #     temp+=x
        #cost= (-1)*temp[0]/m
        #grad= -np.dot(np.transpose(X),(y-self.predict(X)))
        y = np.where( y==0, -1, 1)
        n = len(X)
        a = 0
        for i in range(n):
            a += np.log(1 + np.exp(-y[i] * np.dot(np.transpose(w), X[i])))
        cost = 1/n * a
        
        # Grad
        
        a = 0
        for i in range(n):
            a += (y[i] * X[i])/(1 + np.exp(y[i] * np.dot(np.transpose(w), X[i])))
        grad = - 1/n * a
        ### END CODE
        assert grad.shape == w.shape

        return cost, grad


    def fit(self, X, y, w=None, lr=0.1, batch_size=16, epochs=10):
        """
        Run mini-batch stochastic Gradient Descent for logistic regression 
        use batch_size data points to compute gradient in each step.
    
        The function np.random.permutation may prove useful for shuffling the data before each epoch
        It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
        Remeber the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

        Args:
           X: np.array shape (n,d) dtype float32 - Features 
           y: np.array shape (n,) dtype int32 - Labels 
           w: np.array shape (d,) dtype float32 - Initial parameter vector
           lr: scalar - learning rate for gradient descent
           batch_size: number of elements to use in minibatch
           epochs: Number of scans through the data

        sets: 
           w: numpy array shape (d,) learned weight vector w
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if w is None: w = np.zeros(X.shape[1])
        history = []        
        ### YOUR CODE HERE 14 - 20 lines
        for i in range (epochs):
            # Shuffling the data
            # Resetting the state will ensure the same shuffle
            shuffle_X = np.copy(X)
            shuffle_Y = np.copy(y)
            rng_state = np.random.get_state()
            np.random.shuffle(shuffle_X)
            np.random.set_state(rng_state)
            np.random.shuffle(shuffle_Y)
        
            n = len(shuffle_X)
            b =  math.ceil(n / batch_size)
            batch_index = 0
            for j in range(b):
                batch_end = min(batch_index + batch_size, n)
                batch_X = shuffle_X[batch_index : batch_end]
                batch_Y = shuffle_Y[batch_index : batch_end]
                cost, grad = self.cost_grad(batch_X, batch_Y, w)
                
                w = w - lr * (1 / batch_size) * grad
                history.append(cost)
                
                batch_index += batch_size
            
        ### END CODE
        self.w = w
        self.history = history


    def predict(self, X):
        """ Classify each data element in X

        Args:
            X: np.array shape (n,d) dtype float - Features 
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on X (0, 1)

        """
        pred = np.zeros(X.shape[0])
        ### YOUR CODE HERE 1 - 4 lines
        #z= np.dot(X, self.w)
        #out= logistic(z)
        for i in range(len(X)):
            pred[i] = logistic(np.dot(self.w.T, X[i]))
        
        #creates a list of -1 and 1
        out = np.sign(pred - 0.5)
        
        #converts all -1 to 0, so that out contains 0 and 1 instead
        out = np.where(out==-1, 0,1)
        ### END CODE
        return out
    
    def score(self, X, y):
        """ Compute model accuracy  on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features 
            y: np.array shape (n,) dtype int32 - Labels 

        Returns: 
           s: float, number of correct prediction divivded by n.

        """
        s = 0
        ### YOUR CODE HERE 1 - 4 lines
        s = np.mean(self.predict(X)==y)
        ### END CODE
        return s
        

    
def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([0, 0, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])    
    w = np.array([0.0, 0.0])
    y = np.array([0, 0, 1]).astype('int64')
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    numerical_grad_check(f, w)
    print('Test Success')


    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    
    
