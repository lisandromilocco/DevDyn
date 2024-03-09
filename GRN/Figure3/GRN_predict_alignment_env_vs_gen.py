import copy
import os
import numpy as np
import pylab as pl
from numpy.random import rand
from GRN_development import *
import sys
import scipy.stats


import matplotlib as mpl
import matplotlib.font_manager as fm

n_genes = 5
delta_t = 1
max_steps = int(3e2)
epsilon = 2  # Number of genes that are affected by environmental queues
mut_variance = 0.2
omega=1
var_e = 0.2

pl.rcParams['font.family'] = 'serif'
cmfont = fm.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
pl.rcParams['font.serif'] = cmfont.get_name()
pl.rcParams['mathtext.fontset'] = 'cm'
pl.rcParams['axes.unicode_minus'] = False


def unit_vector(vector):
    # Return unit vector
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

# ----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    seed=0
    np.random.seed(seed)

    folder = "predict_mutation_and_plasticity"
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)


    Theta_store = []

    number_of_references = 100

    x0=np.random.uniform(.1, .1, size=n_genes)
    counter=0

    for counter in range(0, number_of_references):
        good_enough = False
        while good_enough == False:

            B = np.random.normal(0, mut_variance, size=n_genes * n_genes).reshape(n_genes, n_genes)
            M = np.random.randint(2, size=n_genes * n_genes).reshape(n_genes, n_genes)
            Theta = M*B


            # CHECK THE FITNESS OF THE REFERENCE INDIVIDUAL TO MAKE SURE IT IS NOT TOO SMALL

            [fitness, traits] = development_linear_GeneExpressionAsTrait_reference(x0,
                                                    Theta, np.repeat(0, n_genes),np.array((5,5,5,5,5)),
                                                    max_steps, delta_t, n_genes, omega)

            if fitness > 0.005:
                good_enough = True

        Theta_store.append(copy.deepcopy(Theta))

    # Now for each reference, introduce perturbations, simulate them, calculate delta, compare with analytical prediction
    store_deviations_Genetic=[]
    store_deviations_Environmental=[]
    store_angles=[]
    store_angles_random = []
    store_D_lambda=[]

    num_pert=10

    for ref in range(0, number_of_references):
        Theta_ref=Theta_store[ref]
        E_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        KM_ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mu_ref = np.diag([.1, .1, .1, .1, .1])

        reference_traits = development_linear_GeneExpressionAsTrait_Perturbation(x0,Theta_ref, E_ref,
                                                                                 max_steps, delta_t, KM_ref, mu_ref)
        h = np.matmul(Theta_ref, reference_traits) + E_ref
        h[h < 0] = 0

        i=0

        # Modify element j of the i-th row of Theta and introduce a perturbation in i-th environmental factor
        for pert in range(0,num_pert):
           #Genetic perturbation
           Theta = copy.deepcopy(Theta_ref)
           j = np.random.choice(range(0, n_genes))

           delta=np.random.normal(loc=0, scale=mut_variance)
           Theta[i][j] += delta

           perturbed_Genetic = development_linear_GeneExpressionAsTrait_Perturbation(x0, Theta, E_ref,
                                                                                    max_steps, delta_t,
                                                                                    KM_ref, mu_ref)

           if perturbed_Genetic[0]>900: #Filter in the (unlikely) event of lack of convergence
               continue


           # Environmental perturbation
           delta = np.random.normal(loc=0, scale=mut_variance)
           E = copy.deepcopy(E_ref)
           E[i] += delta

           perturbed_Environmental = development_linear_GeneExpressionAsTrait_Perturbation(x0, Theta_ref, E,
                                                                                    max_steps, delta_t,
                                                                                    KM_ref, mu_ref)
           if perturbed_Environmental[0]>900:#Filter in the (unlikely) event of lack of convergence
               continue

           v1 = perturbed_Environmental-reference_traits
           v2 = perturbed_Genetic - reference_traits

            # Angle between perturbations
           angle=np.min([angle_between(v1,v2),angle_between(v1,-v2)])
           store_angles.append(angle)

            # Angle between random vectors in 5-dimensional space
           v1 = np.random.normal(0,1,5)
           v2 = np.random.normal(0,1,5)
           angle_ctrl=np.min([angle_between(v1,v2),angle_between(v1,-v2)])
           store_angles_random.append(angle_ctrl)


    colormap = pl.get_cmap('magma')
    color1 = colormap(0.2)
    color2 = colormap(0.7)

    fig = pl.figure(layout="constrained")
    ax_array = fig.subplots(2, 2, squeeze=False)

    bins=20
    ax_array[0, 0].hist(store_angles, bins, alpha=0.5, color=color1, label="Aligned")#color='crimson')
    ax_array[0, 0].hist(store_angles_random, bins, alpha=0.5, color=color2, label="Random")#='orange')
    ax_array[0, 0].legend(loc='upper right', edgecolor="white")

    for ax in ax_array.flat:
        ax.set(xlabel="Angle")

    #pl.xlabel('Angle')  # Add y-axis label
    pl.savefig(folder + "/angles.pdf",
               format="pdf", bbox_inches="tight")
    pl.close()