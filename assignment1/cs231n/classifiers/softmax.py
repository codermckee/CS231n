import numpy as np
from random import shuffle
from past.builtins import xrange
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_class = W.shape[1]
  num_sample = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_sample):
    Sj = X[i,:].dot(W) #(1,10)
    #temp = Sj - max(Sj)
    #Sj = temp
    #Syi = X[i,:].dot(W[:,y[i]])
    Syi = Sj[y[i]]
    #exp_Syi = np.exp(Syi)
    exp_Sj = np.exp(Sj) #(1,10)
    #loss += -np.log((exp_Syi/np.sum(exp_Sj)))
    loss += (-Syi + np.log(np.sum(exp_Sj)))
    probability = exp_Sj/np.sum(exp_Sj) #(1,10)
    for j in xrange(num_class):
      if j != y[i]:
        dW[:,j] += probability[j] * X[i,:].T #(3073,1)
      else:
        dW[:,j] += (-1 + probability[j]) * X[i,:].T
  dW /= num_sample
  loss /= num_sample
  loss += reg * np.sum(W*W)
  dW += reg * W
     
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_class = W.shape[1]
  num_sample = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W) #(500,10)
  Sy = S[range(num_sample),list(y)] #(500,)
  exp_S = np.exp(S) #(500,10)
  loss = -sum(Sy) + np.sum(np.log(np.sum(exp_S,axis = 1))) 
  P = exp_S/np.sum(exp_S,axis = 1).reshape(-1,1) #(500,10)
  P[range(num_sample),list(y)] += -1
  dW = (X.T).dot(P)
  dW /= num_sample
  dW += reg*W
  loss /= num_sample
  loss += reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

