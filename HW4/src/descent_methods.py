from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import math

mat = loadmat('HW4/data/cvx_hw4_data.mat')
A = mat['A']
b = np.transpose(mat['b'])[0]
c = mat['c'][0][0]

def euclidean_distance(a, b):
    return math.sqrt(sum([math.pow(a_i - b_i, 2) for a_i, b_i in zip(a, b)]))

def quadratic_function(A, q, r):
    return lambda x: 1 / 2 * np.dot(x, np.matmul(A, x)) + np.dot(q, x) + r

    
def gradient_of_quadratic_function(A, q):
    return lambda x: [sum(i) for i in zip(np.matmul(A, x), q)]

def backtracking_line_search(f, grad_f, alpha, beta, step, x):
    t = 1
    while f([sum(j) for j in zip(x, [t * i for i in step])]) > f(x) + alpha * t * np.dot(grad_f(x), step):
        t *= beta
    return t

def steepest_descent(A, q, r, x0, P, e, alpha, beta, MAX_ITERS, title):
    x = x0
    dist_from_opt = float('inf')
    x_optimal = [-1 * i for i in np.matmul(np.linalg.inv(A), q)]
    iters = 0
    x_axis = []
    y_axix = []
    while dist_from_opt > e and iters < MAX_ITERS:
        step = [-i for i in np.matmul(np.linalg.inv(P), gradient_of_quadratic_function(A, q)(x))]
        t = backtracking_line_search(quadratic_function(A, q, r), gradient_of_quadratic_function(A, q), alpha, beta, step, x)
        x = [sum(j) for j in zip(x, [t * i for i in step])]
        iters += 1
        dist_from_opt = euclidean_distance(x, x_optimal)
        x_axis.append(iters)
        y_axix.append(dist_from_opt)
    plt.plot(x_axis, y_axix)
    plt.xlabel("iteration number")
    plt.ylabel("distance from optimal point")
    plt.title(title)
    plt.show()    

steepest_descent(5*np.identity(len(A)), b, c, [0]*len(A), np.identity(len(A)), 0.001, 0.1, 0.8, 100, "A = 5I and gradient descent")
steepest_descent(5*np.identity(len(A)), b, c, [0]*len(A), A, 0.001, 0.1, 0.8, 100, "A = 5I and steepest descent with P=A")
steepest_descent(A, b, c, [0]*len(A), np.identity(len(A)), 0.001, 0.1, 0.8, 100, "A from data.mat and gradient descent")
steepest_descent(A, b, c, [0] * len(A), A, 0.001, 0.1, 0.8, 100, "A from data.mat and steepest descent with P=A")
