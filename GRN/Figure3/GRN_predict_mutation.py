import copy
import os
import numpy as np
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


# ----------------------------------------------------------------------------------------------------------

pl.rcParams['font.family'] = 'serif'
cmfont = fm.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
pl.rcParams['font.serif'] = cmfont.get_name()
pl.rcParams['mathtext.fontset'] = 'cm'
pl.rcParams['axes.unicode_minus'] = False



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
    store_deviations_Simulated=[]
    store_deviations_Analytical=[]

    store_D_lambda=[]

    num_pert=20

    for ref in range(0, number_of_references):
        Theta_ref=Theta_store[ref]
        E_ref = np.array([0, 0, 0, 0, 0])
        KM_ref = np.array([1, 1, 1, 1, 1])
        mu_ref = np.diag([.1, .1, .1, .1, .1])

        reference_traits = development_linear_GeneExpressionAsTrait_Perturbation(x0,Theta_ref, E_ref,
                                                                                 max_steps, delta_t, KM_ref, mu_ref)
        h = np.matmul(Theta_ref, reference_traits) + E_ref
        h[h < 0] = 0

        # Modify KM
        for pert in range(0,num_pert):
           Theta = copy.deepcopy(Theta_ref)

           i = np.random.choice(range(0, n_genes))
           j = np.random.choice(range(0, n_genes))

           # Make sure that the element of theta is not 0 (to avoid a topological change)
           is0=True
           while is0:
               i = np.random.choice(range(0, n_genes))
               j = np.random.choice(range(0, n_genes))
               if Theta_ref[i][j] != 0: is0=False

           delta = np.random.uniform(-Theta_ref[i][j], Theta_ref[i][j], 1)

           Theta[i][j] += delta

           perturbed_traits = development_linear_GeneExpressionAsTrait_Perturbation(x0, Theta, E_ref,
                                                                                    max_steps, delta_t,
                                                                                    KM_ref, mu_ref)

           # Discard in the rare event of no convergence
           if perturbed_traits[0]>900:
               continue

           store_deviations_Simulated.append((perturbed_traits-reference_traits))

           #Now predict the effect of the mutation using the analytical expression of the sensitivity (see Appendix C)
           #Calculate s
           #First the alphas
           alfa=KM_ref/(KM_ref+h)**2
           Alfa = np.diag(alfa)
           #Now A
           A=np.matmul(Alfa,Theta_ref)-mu_ref
           A_inv=np.linalg.inv(A)
           #now b_theta_i_j
           b_theta_i_j=np.array([0,0,0,0,0], dtype=float)
           b_theta_i_j[i]=alfa[i]*reference_traits[j]
           #now s
           s_theta_i_j = -np.matmul(A_inv,b_theta_i_j)
           # now delta lambda
           D_lambda = Theta[i][j]-Theta_ref[i][j]

           # Store prediction
           store_deviations_Analytical.append(s_theta_i_j * D_lambda)
           # Store change in lambda
           store_D_lambda.append(D_lambda/Theta_ref[i][j])



    norm_differences = []
    for i in range(len(store_deviations_Simulated)):
        diff = store_deviations_Simulated[i] - store_deviations_Analytical[i]
        norm = np.linalg.norm(diff)
        norm = norm/np.linalg.norm(store_deviations_Simulated[i])
        norm_differences.append(norm)

        if norm>10:
            print(store_D_lambda[i])

    x_data = np.abs(store_D_lambda)*100
    y_data = np.array(norm_differences)*100

    pl.scatter(x_data,y_data,
                       alpha=.2, marker=".",
               color="black")

    pl.ylabel('Relative error (log)')  # Add x-axis label
    pl.xlabel('Relative size of perturbation (log)')  # Add y-axis label

    #pl.ylim([-.5, .5])

    pl.savefig( folder + "/Dlambda.png")
    pl.close()

    #--------------------------------------------------------------------------
    #x_data=np.log10(np.abs(store_D_lambda))
    #y_data=np.log10(norm_differences)

    binned_stat = scipy.stats.binned_statistic


    def prctile_05(data):
        return np.percentile(data, 10.0)
    def prctile_95(data):
        return np.percentile(data, 90.0)
    def xyBin_md_95ci(xdata, ydata, num_bins):
        # Return xbincenters,ymedian,y05,y95.
        y_md, be, m = binned_stat(xdata, ydata, statistic="median", bins=num_bins)
        y_05, be, m = binned_stat(xdata, ydata, statistic=prctile_05, bins=num_bins)
        y_95, be, m = binned_stat(xdata, ydata, statistic=prctile_95, bins=num_bins)
        x_bins = (be[:-1] + be[1:]) / 2
        return x_bins, y_md, y_05, y_95


    fig = pl.figure(layout="constrained")
    ax_array = fig.subplots(2, 2, squeeze=False)


    ax_array[0, 0].scatter(x_data,y_data,
               alpha=.05, marker=".",
               color="black")
    # now do the calc:
    x1, y1_md, y1_05, y1_95 = xyBin_md_95ci(x_data, y_data, 15)

    colormap = pl.get_cmap('magma')
    color = colormap(0.4)

    # blue data
    #pl.fill_between(x1, y1_05, y1_95, facecolor=color, alpha=0.5, linewidth=0)
    ax_array[0, 0].plot(x1, y1_md, c=color, linewidth=1, linestyle="dashed", alpha= 0.5)

    y_error = [abs(y1_05-y1_md), abs(y1_95-y1_md)]

    ax_array[0, 0].errorbar(x1, y1_md,
                 yerr=y_error,
                 fmt='o', c=color, alpha= 0.5)


    for ax in ax_array.flat:
        ax.set(xlabel=r'Relative size of perturbation (%)',
               ylabel=r'Relative error (%)')
        ax.set_ylim([-1, 100])
        ax.set_xlim([0, 100])

    pl.xlim([0, 100])
    pl.ylim([-1, 100])


    pl.savefig(folder + "/Figure3_A.pdf",
               format="pdf", bbox_inches="tight")

    pl.close()

