'''
	Chi-square pruning on a decision tree.

    We traverse a sklearn DecisionTree to do this

    Example usage:

    # Creating and fitting the classifier.
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)

    # Predicting classes for the test set.
    y_pred = clf.predict(X_test)


    # Initializing the remaining samples.
    rem_samples = {0: np.arange(train_data.shape[0],dtype=int)}

    # Performing the traversal that prunes the tree.
    clf_new = traversal(clf, 0, train_data, rem_samples)

    # Predicting accuracy on the test set with the new classifier.
    y_pred = clf_new.predict(X_test)

'''

import numpy as np
from math import sqrt
from sklearn import tree


# This function goes to the bottom of the tree and prunes
# the decision tree nodes.
def traversal(clf, node_id, train_data,rem_samples, chi_max=10):
    n_nodes = clf.tree_.node_count              
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    clf_new = clf

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        # We generate all the data required to do the the traversal and the chi test.
        curr_feature = feature[node_id]
        curr_threshold = threshold[node_id]
        curr_samples = train_data[rem_samples[node_id]]


        # Remaining samples after decision.
        rem_samples[children_left[node_id]] = np.where(curr_samples[:,curr_feature] <= curr_threshold)
        rem_samples[children_right[node_id]] = np.where(curr_samples[:,curr_feature] > curr_threshold)
    
        # Traversal.
        clf_new = traversal(clf, children_left[node_id], train_data,rem_samples)
        clf_new = traversal(clf_new, children_right[node_id], train_data,rem_samples)

        children_left = clf_new.tree_.children_left
        children_right = clf_new.tree_.children_right


        s = np.count_nonzero(curr_samples[:,-1] )
        p = np.count_nonzero(curr_samples[:,-1] > 0)

        s_T = np.count_nonzero(curr_samples[:,curr_feature] <= curr_threshold)
        s_F = np.count_nonzero(curr_samples[:,curr_feature] > curr_threshold)
        p_T = np.count_nonzero(curr_samples[:,curr_feature] <= curr_threshold * curr_samples[:,-1] )
        p_F = np.count_nonzero(curr_samples[:,curr_feature] * curr_samples[:,-1]  > curr_threshold)

        # If right child is a leaf.
        if children_right[node_id] > 0 and children_left[node_id] > 0:
            if (children_right[children_right[node_id]] == children_left[children_right[node_id]]):
                # Do the chi test.
                if s * s_F *s_T * p > 0:
                    chi = ( (p_F - float(s_F * p) / s) * (p_F - float(s_F * p) / s)  / (float(s_F * p / s)) ) + ( (p_T - float(s_T * p / s)) * (p_T - float(s_T * p / s)) / float(s_T * p / s) )
                    #print node_id
                    print chi
                    if chi > chi_max:
                        #print "yes"
                        clf_new.tree_.children_right[node_id] = -1
                        clf_new.tree_.children_left[node_id] = -1

        # # If left child is a leaf.
        # if children_left[node_id] > 0:
        #     if (children_right[children_left[node_id]] == children_left[children_left[node_id]]):
        #         # Do the chi test.
        #         if s * s_F * s_T * p > 0:
        #             chi = (p_F - float(s_F * p) / s) * (p_F - float(s_F * p) / s)  / (float(s_F * p / s)) + (p_T - float(s_T * p / s)) * (p_T - float(s_T * p / s)) / float(s_T * p / s)
        #             print "l"
        #             print node_id
        #             print chi
        #             if chi > chi_max:
        #                 children_left[node_id] = -1

    # Return the new classifier.
    return clf_new



