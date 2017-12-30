'''
    Decision trees optimizing over weight misclassification rate.

    # Building the decision tree.
    tr = buildtree(train_data, 7)

    # Classify test_set based on decision tree.
    y_pred = predict(X_test, tr)

    To Do: Clean up
'''
import operator

import numpy as np
from scipy import stats



# The decision node class
class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col            # Index of attribute tested.
        self.value=value        # Threshold value.
        self.results=results    # dict of results for a branch.
        self.tb=tb              # true decision nodes 
        self.fb=fb              # false decision nodes


# Divides a based on a specific column with a threshold value. 
def divide_set(data,column,t):
   set_T= data[data[:,column] <= t]
   set_F= data[data[:,column] > t]

   return (set_T,set_F)

# Calculates the weight misclassification error.
def weight_misclass(samples, a, t):
	divided_set = divide_set(samples,a,t)
	p_below = divided_set[0].shape[0] / float(samples.shape[0])
	p_above = divided_set[1].shape[0] / float(samples.shape[0])

	l_below = stats.mode(divided_set[0])[-1]
	l_above = stats.mode(divided_set[1])[-1]

	res = p_above * np.count_nonzero(divided_set[1][:,-1] != l_above) + p_below * np.count_nonzero(divided_set[0][:,-1] != l_below)
	return res

# Finds counts of each class for given dataset.
def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results

# Builds a tree
def buildtree(rows, max_depth):
    # If maximum depth reached, return.
    if max_depth < 0: return decisionnode(results=uniquecounts(rows))

    # If empty set, return.
    if len(rows) == 0: return decisionnode()

    best_err = float("inf")
    best_criteria = None
    best_sets = None
    best_split = "inf"

    # For all columns
    n_cols = len(rows[0]) - 1	  # Last column is result

    # For each column.
    for col in range(0, n_cols):
        # Find different values possible in this column
        column_values = set([row[col] for row in rows])

        # For each possible value, try to divide on that value
        for value in column_values:
            set_T, set_F = divide_set(rows, col, value)

            # Check the misclassification error.
            err = weight_misclass(rows, col, value)
            
            # If the error is better, update the best attributes.
            if (err < best_err or (err == best_err and abs(len(set_T) - len(set_F)) < best_split) ) and len(set_T) > 0 and len(set_F) > 0:
                best_err = err
                best_criteria = (col, value)
                best_sets = (set_T, set_F)
                best_split = abs(len(set_T) - len(set_F))

    # Now, we summarize the results coming out of this node.
    res=uniquecounts(rows) 

    # If still need to divide the samples into classes.
    if best_err < "inf" and len(res) == 2 and best_sets != None and best_sets[0] != None and best_sets[1] != None:
    	
        # Build the branchs recursively.
        trueBranch = buildtree(best_sets[0], max_depth - 1)
        falseBranch = buildtree(best_sets[1], max_depth - 1)

        # Return the tree created.
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                tb=trueBranch, fb=falseBranch, results=res)

    # If the samples reaching here are purely of one class, go no further.
    else:
        return decisionnode(results=res)

# Traverses the tree to obtain the predicted class.
def predict(test_set, tree):
	y_pred = []

    # We predict each row.
	for row in test_set:
		curr_node = tree
		prev_node = None
        
		while(curr_node != None):
			prev_node = curr_node
			if (row[curr_node.col] <= curr_node.value):
				curr_node = curr_node.tb
			else:
				curr_node = curr_node.fb

        # Take majority vote.
		y_pred.append(max(prev_node.results.iteritems(), key=operator.itemgetter(1))[0])

	return y_pred


