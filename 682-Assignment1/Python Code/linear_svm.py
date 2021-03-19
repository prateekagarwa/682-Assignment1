import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]  #10
  num_train = X.shape[0]   #minibatch size 
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      #count the number of classes which are greater than 0 for updating grad y_i
      if margin > 0:
        loss += margin
        dW[:,j]+= X[i]
        dW[:,y[i]]+= -1*X[i]
  
  dW= np.divide(dW,num_train)
  # add the regularizer 
  dW=dW+ 2*reg*W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)



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
  num_train= X.shape[0]
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
  loss= X @ W
  #subtract score of correct class from each row and add 1
  # L=L-  np.diagonal( W[:, y])[:,None]
  loss=loss- loss[np.arange(loss.shape[0]),y][:,None]
  # add 1 to every element but the y_i column
  loss=loss+1
  loss[np.arange(loss.shape[0]),y]=0
  #subtract 1 from the y column, if less than 0 become 0
  loss[loss < 0 ]=0
  gradloss= np.copy(loss)
  
  
  loss=np.sum(loss)
  loss=loss/X.shape[0]+ reg* np.sum(np.square(W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

#vectorized version = X.T which is D*N @ L which is N* C and encodes vector mult and gives D*C which is dW size
  gradloss[gradloss>0]=1
  y_value= -1* np.sum(gradloss,axis=1)
  gradloss2 = np.zeros(gradloss.shape)
  gradloss2[np.arange(gradloss.shape[0]),y]=1
  gradloss2= gradloss2*y_value[:,np.newaxis]
  dW= np.divide(X.T @ gradloss + X.T@ gradloss2,num_train) + 2*reg*W
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
 














  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
