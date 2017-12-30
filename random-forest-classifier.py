'''
	Implementation of Random Forests using 
	Scikitlearn's Decision Trees.


	Example usage:
		rfc = MyRandomForestClassifier(10, "auto", X_train, y_train)

		my_y_pred = rfc_predict(X_test, rfc)
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


# On input of the parameters, gives the list of classifiers.
def MyRandomForestClassifier(k, m, X_train, y_train):
	rfc = []
	n = X_train.shape[0]

	for i in range(0, k):
		subsample_indices = np.random.random_integers(0, n - 1, size=(n,))

		# Taking the subset.
		X_train_sub = X_train[subsample_indices,:]
		y_train_sub = y_train[subsample_indices]

		# Classifier and fitting it. Consider m features per node.
		clf = tree.DecisionTreeClassifier(max_features=m)
		rfc.append(clf.fit(X_train_sub, y_train_sub))

	return rfc

# Predicts the output given a Random Forest Classifiers.
def rfc_predict(X_test, rfc):
	y_pred = np.zeros((X_test.shape[0],))

	# The predictions.
	y_preds = []
	for clf in rfc:
		y_preds.append(clf.predict(X_test))

	for i, sample in enumerate(X_test):
		count = {0:0, 1:0}
		for preds in y_preds:
			count[preds[i]] += 1
		if count[1] > count[0]:
			y_pred[i] = 1

	return y_pred



