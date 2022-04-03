# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:52:42 2022

@author: carlo
"""
# Hungarian Algo for Bipartite graphs:
    # return the minimal cost matching 
    # 1) with adiajency matrix: suitable for dense graph, library scipy.optimize.linear_sum_assignment
    # 2) with graph: suitable for sparse graph, library NetworkX
    
# adiajency matrix reports the weight of the edge connecting right and left side nodes

import numpy as np
from scipy.optimize import linear_sum_assignment


# STEP 1: subtract from each row of the matrix the smallest element in that row
def step1(m):
    for i in range(m.shape[0]):
        m[i,:] = m[i,:] - np.min(m[i,:])
        

# STEP 2: subtract from each column of the matrix the smallest element in that column
def step2(m):
    for j in range(m.shape[1]):
        m[:,j] = m[:,j] - np.min(m[:,j])
        

# STEP 3: find minimal number of lines s.t. I pass over all zeros of the matrix
def step3(m):
    dim = np.shape(m)[0]
    assigned = np.array([])                       # list to collect indexes of assigned rows
    assignments = np.zeros(m.shape, dtype=int)    # matrix to store the assigments, (i,j) has value 1 if row i is assigned to column j
    
    # iterate over rows and columns, if we find a zero and rows and cols has not been assigned yet, assign
    for i in range(0, dim):
        for j in range(0, dim):
            if (m[i,j] == 0 and np.sum(assignments[:,j]) == 0 and np.sum(assignments[i,:]) == 0):
                assignments[i,j] = 1
                assigned = np.append(assigned,i)   # store indexes of the rows assigned
                
    # we use assigned rows to initialize the marked rows variable i.e. the non assigned rows
    rows = np.linspace(0, dim-1, dim).astype(int)
    marked_rows = np.setdiff1d(rows, assigned)
    new_marked_rows = marked_rows.copy()
    marked_cols = np.array([])
    
    
    while (len(new_marked_rows) > 0):               # iterate until we have new marked rows
        new_marked_cols = np.array([], dtype=int)
        for nr in new_marked_rows:                  # iterate over the new marked rows
            zeros_cols = np.argwhere(m[nr,:]==0).reshape(-1)    # mark columns that have zero in the current row
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols))   # add them to new marked columns (discarding the columns we already marked before with set_diff)
        marked_cols = np.append(marked_cols, new_marked_cols)   # add new marked columns in the set of marked columns
        new_marked_rows = np.array([], dtype=int)
    
        # iterate over the new marked columns
        for nc in new_marked_cols:
            # update values of new marked rows appending indexes of rows that have an assignment in the current column
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:,nc]==1).reshape(-1))
        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))
    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)
    # return indexes of the unmarked rows and the indexes of the marked columns
    
    
# STEP 4: check if the number of lines we draw is equal to the number or rows or columns
# if this is true, we can find an optimal assignment for the zeros and the algos ends, otherwise go to step 5

    
# STEP 5: find the smallest entry not covered by any line. Then subtract this entry from each row that
# is not crossed out and add it to each column that is crossed out; In this way I add one more zero to the matrix
# then, go to step 3
def step5(m, covered_rows, covered_cols):
    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0]-1, m.shape[0]), covered_rows).astype(int)   # rows not covered by lines
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[0]-1, m.shape[0]), covered_cols).astype(int)   # cols not covered by lines
    min_val = np.max(m)                      # minimum value of the matrix
    for i in uncovered_rows.astype(int):     # find minumum value among uncovered units
        for j in uncovered_cols.astype(int):
            if m[i,j] < min_val:
                min_val = m[i,j]
    for i in uncovered_rows.astype(int):     # substact minimum value from uncovered rows
        m[i,j] -= min_val
    for j in covered_rows.astype(int):       # add minimum value to covered columns
        m[i,j] += min_val
    return m
    

# returns the first row with a zero, if does not find return false
def find_rows_single_zero(matrix):
    for i in range(0, matrix.shape[0]):
        if np.sum(matrix[i,:] == 0) == 1:
            j  = np.argwhere(matrix[i,:]==0).reshape(-1)[0]
            return i,j
    return False

# returns the first column with a zero, if does not find return false
def find_cols_single_zero(matrix):
    for j in range(0, matrix.shape[1]):
        if np.sum(matrix[:,j] == 0) == 1:
            i  = np.argwhere(matrix[:,j]==0).reshape(-1)[0]
            return i,j    # attenzione, nel codice dell'esercitazione li inverte
    return False


  
# implement a function that only computes the assigments for rows and columns with a single zero
def assignment_single_zero_lines(m, assignment):
    val = find_rows_single_zero(m)
    while(val):
        i,j = val[0], val[1]
        m[i,j] += 1
        m[:,j] += 1
        assignment[i,j] = 1
        val = find_rows_single_zero(m)
        
    val = find_cols_single_zero(m)
    while(val):
        i,j = val[0], val[1]
        m[i,:] += 1
        m[i,j] += 1
        assignment[i,j] = 1
        val = find_cols_single_zero(m) 
    
    return assignment  # return a partial assignment assigning only rows and cols with a single zero


# return the indexes of the zero covered by a line with more than one zero
def first_zero(m):
    return np.argwhere(m==0)[0][0], np.argwhere(m==0)[0][1]



def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype=int)
    assignment = assignment_single_zero_lines(m, assignment)
    while np.sum(m==0) > 0:    # we look for rows and columns with at least two zeros
        i,j = first_zero(m)
        assignment[i,j] = 1
        m[i,:] += 1
        m[:,j] += 1
        assignment = assignment_single_zero_lines(m, assignment)
    return assignment*initial_matrix, assignment
        
        
# function that puts together all the functions     
def hungarian_algorithm(matrix):
    m = matrix.copy()
    step1(m)
    step2(m)
    n_lines = 0   # store number of lines found
    max_length = np.maximum(m.shape[0], m.shape[1])
    while n_lines != max_length:
        lines = step3(m)   # return lines found
        n_lines = len(lines[0]) + len(lines[1])
        if n_lines != max_length:
            step5(m, lines[0], lines[1])
    return final_assignment(matrix, m)
       

# initialize an adiagency matrix
a = np.random.randint(100, size=(3,3))

res = hungarian_algorithm(a)
print("\n Optimal Matching: \n", res[1], "\n Value: ", np.sum(res[0]))
        
        
    
    
    
    
    