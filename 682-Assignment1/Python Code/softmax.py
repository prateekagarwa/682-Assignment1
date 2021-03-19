import numpy as np
from random import shuffle
import math as math

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
  num_classes = W.shape[1]  #10
  num_train = X.shape[0]   #minibatch size
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape)

  for i in range(num_train):
    scores=X[i].dot(W)
    
    #make the largest score =0, to avoid numerical instability
    scores= np.subtract(scores,np.amax(scores))
    scores= np.exp(scores)
    # top= scores[y[i]]
    bottom=np.sum(scores)
    scores=scores/bottom
    loss+= -1* math.log(scores[y[i]])

    for j in range(num_classes):
      if j == y[i]:
        dW[:,j]+= X[i]* (scores[j]-1)
      else:
        dW[:,j]+= X[i]* (scores[j])



  dW= np.divide(dW,num_train)
  # add the regularizer 
  dW=dW+ 2*reg*W
  
  loss= loss/num_train
  loss+= reg * np.sum(W * W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
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
  dW = np.zeros(W.shape)
  num_train= X.shape[0]

  loss=X@W
  #subtract maximum from every row
  loss= loss-np.max(loss,axis=1)[:,None]
  loss=np.exp(loss)
  sums= np.sum(loss,axis=1)
  loss= loss/ sums[:,None]
  gradloss=np.copy(loss)
  loss= -1* np.log(loss)

  sum_y_values= np.sum( loss[np.arange(loss.shape[0]),y])
  sum_y_values=(sum_y_values/num_train)+ reg* np.sum(np.square(W))
  loss=sum_y_values

  gradloss_yvals= gradloss[np.arange(gradloss.shape[0]),y]
  gradloss_yvals= gradloss_yvals+ (-1)
  
  gradloss[np.arange(gradloss.shape[0]),y]=gradloss_yvals
  # gradloss=gradloss/num_train
  
  dW=  (X.T@gradloss)/num_train +2*reg*W






  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

