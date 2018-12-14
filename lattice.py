
# coding: utf-8

# In[6]:

import numpy as np
import math
# Boundary types:
# type 0: there is no sink
# type 1: only the corners are connected to a sink by a single edge
# type 2: all boundary vertices are connected to a sink through a single edge
# type 3: all noncorner boundary vertices are connected to a sink through a single edge,
#           and each corner is connected to the sink through two edges
# type 4: (for linear lattices) no global sink, yet each endpoint is connected to the other through a directed edge.
# type 5: (when lattice size is >= 2) each vertex on the far left is conected by a single edge to the corresponding vertex on the far right. 
#
# The third argument in each list is the number of edges if the vertex is on both the top/bottom and left/right,
# the second is the number of edges if the vertex is on either the top/bottom or the left/right but not both,
# and the first is the number of edges if the vertex is neither on the top/bottom or the left/right.
boundary_types = {
     0: [0, 0, 0],
     1: [0, 0, 1],
     2: [0, 1, 1],
     3: [0, 1, 2]
};


# In[7]:

# I number the vertices in the lattice from 0 to mn-1, increasing from left to right and 
# moving to the start of the next row as the end of a row is reached (the lattice has m
# rows and n columns).
def adjacency_matrix(m,n,boundarytype):
    vertices = np.reshape(np.arange(0,m*n),(m,n));
    adj_matrix = [];
    for i in range(0,m*n):
        row = [];
        for j in range(0,m*n):
            if (((i == j+1 or i == j-1) and max(i,j)%n != 0) or abs(i-j) == n): row.append(1);
            else: row.append(0);
        adj_matrix.append(row);

    if (boundarytype == 4):
        if (n == 1 and m == 1): adj_matrix[0][0]=1 #self loop in the case of a single vertex 
        elif (n == 1 and m != 1):
            adj_matrix[0][m-1] = 1
            adj_matrix[m-1][0] = 1
        elif (n != 1 and m == 1):
            adj_matrix[0][n-1] = 1
            adj_matrix[n-1][0] = 1

    if (boundarytype == 5):
        if (n == 1 and m == 1): adj_matrix[0][0]=1 #self loop in the case of a single vertex 
        else:
            for i in range(0, m):
                adj_matrix[n*i][n*i+n-1]=1
                adj_matrix[n*i+n-1][n*i]=1

    return np.array(adj_matrix);


# In[8]:

# If calling with boundary type 4, the graph must be linear!
def degree_matrix(m,n,boundarytype):
    deg_matrix = [];
    for i in range(0,m*n):
        row = [];
        for j in range(0,m*n):
            if (i == j):
                row.append(num_neighbors(i,m,n,boundarytype));
            else: row.append(0);
        deg_matrix.append(row);
    return np.array(deg_matrix);


# In[9]:

# i is the vertex number (the vertices in the lattice are numbered from 0 to mn-1, increasing 
# from left to right and moving to the start of the next row as the end of a row is reached.
def num_neighbors(i,m,n,boundarytype):
        if (n==1 and m==1): return(1) #self loop in the case of a single vertex
        
        ontoporbottom = (0 <= i < n or 0 <= m*n-1 - i < n)
        onleftorright = (i%n == 0 or i%n == n-1);

        if boundarytype == 4:
            if (n!=1 and m!=1): return(-1)
            else: return(2)
            
        if boundarytype == 5:
            if (i == 0 or i == m*n-n or i == n-1 or i == m*n-1): return(3)
            elif (onleftorright): return(4)
            elif (ontoporbottom): return(3)
            else: return(4)

        edgestosink = boundary_types[boundarytype][ontoporbottom+onleftorright];

        #if boundarytype != 0 and (ontoporbottom or onleftorright):
        #    if (ontoporbottom and onleftorright): edgestosink = 1 if boundarytype==1 or boundarytype==2 else 2;
        #    else: edgestosink = 0 if boundarytype==1 else 1;

        if (n==1 or m==1):
            startingval = 3
        else:
            startingval = 4

        return(startingval - (ontoporbottom + onleftorright) + edgestosink);


# In[10]:

# Iterates the given sandpile configuration simulataneously multiple times (on each iteration firing all vertices that can be fired), returning the stabilization (the resulting stable configuration) if it occurs and -1 if it does not.
def iterate(m,n, sandpile_configuration, boundarytype=0, num_iterations=10, verbose_mode=True):
    for k in range(0,num_iterations):
        if (verbose_mode): print("starting iteration with configuration:")
        if (verbose_mode): print(np.reshape(sandpile_configuration,(m,n)))
        verticestotopple = []; 
        deg_matrix = degree_matrix(m,n,boundarytype);
        #print("degree matrix")
        #print(deg_matrix)
        adj_matrix = adjacency_matrix(m,n,boundarytype)
        #print("adjacency matrix")
        #print(adj_matrix)
        reduced_laplacian = np.subtract(deg_matrix, adj_matrix)
        for i in range(0, m*n):
            if (sandpile_configuration[i] >= num_neighbors(i,m,n,boundarytype)):
                verticestotopple.append(i);
        if (len(verticestotopple) == 0):
            if (verbose_mode): print("no vertices can be fired: the configuration is stable")
            return sandpile_configuration;
        for k in range(0, len(verticestotopple)):
            i = verticestotopple[k]
            sandpile_configuration = np.subtract(sandpile_configuration, reduced_laplacian[i]);
            if (verbose_mode): print("firing vertex number",i, "gives the configuration:");
            if (verbose_mode): print(np.reshape(sandpile_configuration,(m,n)));
        if (verbose_mode): print("iteration completed")
    return(-1);


# In[11]:

# Computes the stabilization of the given sandpile configuration sigma on an LxL square graph with boundary
# type 3 (see the top of the notebook for a specification of the boundary types) (namely, every vertex on 
# the graph has degree 4, if it can be reached within 50000 iterations!
#
# One should only call this method by passing in a parameter sigma which is a nonempty numpy array, whose length
# is n^2, that represents the initial sandpile configuration on a square grid with side length n. For instance
# if you call this method as follows:
#      'stab(np.array([6,1,4,3,2,7,15,12,18,14,0,0,1,2,5,19]))'
# the method will interpret the given array as the following sandpile configuration on a 4x4 graph:
#      [6  1  4   3]
#      [2  7  15  12]
#      [18 14  0  0]
#      [1  2  5  19]
def stab(sigma):
    L = int(round(math.sqrt(len(sigma))));
    return iterate(L,L, sigma, 3, 50000, False);


# In[25]:

# Examples: 
print(stab(np.array([1,5,1,1])))
print("\n")
iterate(2,2,np.array([0,1,5,2]),3,10,True)
print("\n")
iterate(1,10,np.array([0,6,0,4,0,2,0,8,0,2]),4,10,True)
print("\n")

# When iterate is called as is done in the below cell, it iterates a sandpile configuration that looks like:
#  0 6 0 4
#  0 2 0 4
#  0 2 0 1
iterate(3,4,np.array([0,6,0,4,0,2,0,4,0,2,0,1]),2,10,True)


# In[33]:

init_config_vector = np.array([0,5,4,2,0,6,7,0,4,7,8,0,8,6,4,0,6,7,0,8,7,1,3,4,0,4,0,3,0,3,0,6,7,4,2,2])
grid_size = int(round(np.sqrt(len(init_config_vector))))
init_config = np.reshape(init_config_vector,(grid_size,grid_size))
print("Initial configuration:")
print(init_config)
stab_config_vector = stab(init_config_vector)
stab_config = np.reshape(stab_config_vector,(grid_size,grid_size))
print("Stabilization:")
print(stab_config)


# In[90]:

# The paper (https://arxiv.org/pdf/0801.3306.pdf) claims that the identity configuration of the sandpile group on a finite digraph # with a global sink is I = stab(σ - stab(σ)), where σ = 2𝛿 - 2 and 𝛿(v) = d_v for all v. What is conjectured but not proven
# is that for any L, considering the square grid graph of size L where each noncorner boundary vertex is connected to a global 
# sink through a single edge and each corner vertex is connected to the sink through two edges, the identity configuration
# of the sandpile group for this graph is uniformly 2 (except on the lines through the center of the graph) in a square
# region around the center of the graph.

# Computes the identity in the case that L=15
L=15
delta = np.array([4]*(L**2))
sigma = 2*delta - 2
stab_sigma = stab(sigma)
identity_config = np.reshape(stab(sigma - stab_sigma),(L,L))
print("Identity for L = " + str(L) + ":")
print(identity_config)


# In[75]:

def list_add(A, B):
    ans = []
    if len(A) == len(B):
        for i in range(len(A)):
            ans = ans + [A[i] + B[i]]
        return ans
    else:
        print("Cannot pointwise add lists with different sizes. ")
        return -1

# computes the difference between matrices A and B (represented as numpy arrays)
def diff(A, B):
    return(A-B)

def epsilon(m,n,boundarytype,num_it):
    """computes, 2 * delta - stab(2 * delta) (where delta(v) = outdeg(v) for all vertices v)"""
    deg_config = []
    for i in range(m*n):
        deg_config = deg_config + [num_neighbors(i,m,n,boundarytype)]  # Find degree configuration
    config = list(map((lambda x: 2*x), deg_config))  # Multiply this by two
    stab_config = iterate(m,n,config,boundarytype,num_it,False)
    return diff(config,stab_config)

def recurrent_rep(arr,m,n,boundarytype,num_it):
    """finds the unique recurrent chip config in the same equivalence class as array 
        (see lemma 2.14 in https://arxiv.org/pdf/0801.3306.pdf)"""
    ep = epsilon(m,n,boundarytype,num_it)
    config = list_add(arr,ep)
    return iterate(m,n,config,boundarytype,num_it,False)


# In[65]:

epsilon(3,6,3,100)


# In[87]:

np.reshape(recurrent_rep([0]*12,3,4,3,100),(3,4))


# In[94]:

# Since the identity configuration of the group of recurrent configurations 
# is in the same equivalence class as the zero configuration, we can use this
# method to compute the identity configuration.

# Identity configuration for the group on the square 15x15 lattice with boundary type 3: 
print(np.reshape(recurrent_rep([0]*225,15,15,3,400),(15,15)))


# In[ ]:



