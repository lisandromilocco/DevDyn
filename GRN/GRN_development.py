import matplotlib.pyplot as pl
import matplotlib
matplotlib.use('Agg')
import os
import pickle
from itertools import product
import sys; sys.path.insert(0, "../..")
import numpy as np
import multiprocessing
from multiprocessing import Pool
import time


from scipy.stats import wasserstein_distance


def update_gene_expression(gene_expression, MB, E, delta_t):
    mu=0.1
    KM=1

    R = np.matmul(MB, gene_expression) + E
    R [ R < 0 ] = 0
    diff = (R/(KM+R) - mu * gene_expression) * delta_t

    gene_expression += diff

    return gene_expression

def update_gene_expression_perturbation (gene_expression, MB, E, delta_t, KM, mu):

    R = np.matmul(MB, gene_expression) + E
    R [ R < 0 ] = 0
    diff = (R/(KM+R) - 0.1 * gene_expression) * delta_t

    gene_expression += diff


    return gene_expression



def development_perturbation (initial_condition, M, B, E, W, Z, folder,
                 generation, individual, environment, max_steps, delta_t,
                 n_genes):

    MB = M*B
    gene_expression = initial_condition.copy()

    store_gene_expression=[]
    #####################################################################################
    Stable = True
    for t in range(max_steps):
        store_gene_expression.append(gene_expression.copy())
        gene_expression_old = gene_expression.copy()
        gene_expression = update_gene_expression(gene_expression, MB, E, delta_t)

        if ((t+1) % 40 == 0):
            if np.linalg.norm((gene_expression - store_gene_expression[-40])/
                              np.max(store_gene_expression)) < 1e-2:
                break
        if (t == (max_steps-1)):
            if np.linalg.norm((gene_expression - store_gene_expression[-40]) /
                              np.max(store_gene_expression)) > 1e-2:
                Stable = False
                #print("Not stable")
    #####################################################################################

    #traits = np.array((0.0,0.0))
    traits = np.matmul(gene_expression,(W * Z))
    traits = np.array((gene_expression[2], gene_expression[3]))

    if not Stable:
        traits=np.array((-1,4))

    save_data = [traits, E]


    with open(folder + "/perturbation/individual_" + str(individual) +
              "_environment"+ str(environment) +".pkl",'wb') as filename:
           pickle.dump(save_data, filename)


        #####################################################################################




def development_linear (initial_condition, M, B, E, W, Z, folder,
                 generation, individual, optimum, max_steps, delta_t,
                 n_genes):

    MB = M*B
    gene_expression = initial_condition.copy()

    store_gene_expression=[]
    #####################################################################################
    Stable = True
    for t in range(max_steps):
        store_gene_expression.append(gene_expression.copy())
        gene_expression_old = gene_expression.copy()
        gene_expression = update_gene_expression(gene_expression, MB, E, delta_t)

        if ((t+1) % 40 == 0):
            if np.linalg.norm((gene_expression - store_gene_expression[-40])/
                              np.max(store_gene_expression)) < 1e-2:
                break
        if (t == (max_steps-1)):
            if np.linalg.norm((gene_expression - store_gene_expression[-40]) /
                              np.max(store_gene_expression)) > 1e-2:
                Stable = False
                #print("Not stable")
    #####################################################################################

    #traits = np.array((0.0,0.0))
    traits = np.matmul(gene_expression,(W * Z))
    #traits = np.array((gene_expression[0], gene_expression[1]))

    # Assign fitness
    if not Stable:
        fitness = 0
    else:
        distance_to_opt = np.sqrt(sum((traits-optimum)**2))
        #fitness= np.exp(-5*(distance_to_opt**2)/2)
        fitness = np.exp(-10/8 * (distance_to_opt ** 2) / 2)
        fitness = np.exp(-.5 * (distance_to_opt ** 2) / 2)

    return [fitness, traits]


def development_linear_GeneExpressionAsTrait(initial_condition, M, B, E, W, Z, folder,
                 generation, individual, optimum, max_steps, delta_t,
                 n_genes,omega):

    MB = M*B
    gene_expression = initial_condition.copy()

    store_gene_expression=[]
    #####################################################################################
    Stable = True
    for t in range(max_steps):
        store_gene_expression.append(gene_expression.copy())
        gene_expression_old = gene_expression.copy()
        gene_expression = update_gene_expression(gene_expression, MB, E, delta_t)

        if ((t+1) % 40 == 0):
            if np.linalg.norm((gene_expression - store_gene_expression[-40])/
                              np.max(store_gene_expression)) < 1e-2:
                break
        if (t == (max_steps-1)):
            if np.linalg.norm((gene_expression - store_gene_expression[-40]) /
                              np.max(store_gene_expression)) > 1e-2:
                Stable = False


    traits = np.array((gene_expression[2], gene_expression[3]))

    # Assign fitness
    if not Stable:
        fitness = 0
    else:
        distance_to_opt = np.sqrt(sum((traits-optimum)**2))
        fitness = np.exp(-omega * (distance_to_opt ** 2) / 2)

    return [fitness, traits]




#######################################################################################
#######################################################################################
# For GRN_predict_mutation.py and GRN_predict_alignment_env_vs_gen.py

def development_linear_GeneExpressionAsTrait_reference(initial_condition, Theta, E,
                                                       optimum, max_steps, delta_t,
                 n_genes,omega):

    MB=Theta
    gene_expression = initial_condition.copy()

    store_gene_expression=[]
    #####################################################################################
    Stable = True
    for t in range(max_steps):
        store_gene_expression.append(gene_expression.copy())
        gene_expression_old = gene_expression.copy()
        gene_expression = update_gene_expression(gene_expression, MB, E, delta_t)

        if ((t+1) % 40 == 0):
            if np.linalg.norm((gene_expression - store_gene_expression[-40])/
                              np.max(store_gene_expression)) < 1e-2:
                break
        if (t == (max_steps-1)):
            if np.linalg.norm((gene_expression - store_gene_expression[-40]) /
                              np.max(store_gene_expression)) > 1e-2:
                Stable = False
                #print("Not stable")
    #####################################################################################

    traits = gene_expression


    # Assign fitness
    if not Stable:
        fitness = 0
    else:
        distance_to_opt = np.sqrt(sum((traits-optimum)**2))
        fitness = np.exp(-omega * (distance_to_opt ** 2) / 2)


    return [fitness, traits]


def development_linear_GeneExpressionAsTrait_Perturbation(initial_condition, Theta, E,
                                                          max_steps, delta_t, KM, mu):

    MB = Theta
    gene_expression = initial_condition.copy()

    store_gene_expression=[]
    #####################################################################################
    Stable = True
    for t in range(max_steps):
        store_gene_expression.append(gene_expression.copy())
        gene_expression_old = gene_expression.copy()
        gene_expression = update_gene_expression_perturbation(gene_expression, MB, E, delta_t, KM, mu)

        if ((t+1) % 40 == 0):
            if np.linalg.norm((gene_expression - store_gene_expression[-40])/
                              np.max(store_gene_expression)) < 1e-2:
                break
        if (t == (max_steps-1)):
            if np.linalg.norm((gene_expression - store_gene_expression[-40]) /
                              np.max(store_gene_expression)) > 1e-2:
                Stable = False
                #print("Not stable")
    #####################################################################################

    if Stable:
        traits = gene_expression
    else:
        traits = np.array([999,999,999,999,999])

    return traits