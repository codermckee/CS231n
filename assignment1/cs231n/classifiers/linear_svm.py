# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero (3073,10)

  # compute the loss and the gradient
  num_classes = W.shape[1]    #10类
  num_train = X.shape[0]      # 500个训练样本
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)		# (1,10)
    correct_class_score = scores[y[i]] #第y[i]类的得分 或 第i个样本的所属类别的得分
    for j in xrange(num_classes):
      if j == y[i]:
        continue			#跳出当前循环，执行下一次循环
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin			#计算data loss
        dW[:,j] += X[i].T       # dW: 3073x10, 此处的值为3073x1,放在dW第j列，dW[:,j]只需加一次，外循环再循环500次
        dW[:,y[i]] += -X[i].T	#dW[:,y[i]要加9次]，外循环再循环500次
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train			#由于样本太多，loss值累积，所以取均值
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)		# L2 regularization
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct = []
  for i in xrange(X.shape[0]):
    correct.append(scores[i,y[i]])
  #或： correct = scores[range(500),list(y)]
  correct = np.mat(correct).T
  margin = np.maximum(0,scores - correct + 1) 
  loss = np.sum(margin) + np.sum(correct) - X.shape[0]  #将上面naive中j==y[i]多减多加的部分还原
  #处理2：也可以将j==y[i]的项置0
  #即：margin[range(500),list(y)]=0
  loss = loss/X.shape[0]
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  coeff = np.zeros(margin.shape) #500x10
  coeff[margin>0]=1
  coeff[range(margin.shape[0]),list(y)] = 0 #将j==y[i]的项=0
  coeff[range(margin.shape[0]),list(y)] = -np.sum(coeff,axis=1) #等号后面是(500,)
  dW = (X.T).dot(coeff)#可根据naive解释
  dW = dW/margin.shape[0] + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
