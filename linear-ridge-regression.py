'''
	Linear ridge regression 
	
	Example Usage:
		w = mylinridgereg(X_train, y_train, l)

		y_train_pred = mylinridgeregeval(X_train, w)
		y_pred = mylinridgeregeval(X_test, w)

		mse_train = meansquarederr(y_train, y_train_pred)
		mse_test = meansquarederr(y_test, y_pred)
'''

import numpy as np



# Calculates the linear least squares solution with the ridge
# regression penalty parameter (λ). Returns the regression weights.
def mylinridgereg(X, y, l):
	# w = (X'X + lI)^-1X'y
	X_t = np.transpose(X)
	term1 = np.linalg.inv(X_t.dot(X) + l * np.identity(X.shape[1]))
	w = term1.dot(X_t.dot(y))

	return w 

# Returns the predictions given the weights.
def mylinridgeregeval(X, weights):
	return X.dot(weights)

# Mean squared error between predicted and actual target values.
def meansquarederr(T, Tdash):
	return (np.square(T - Tdash)).mean()

	
