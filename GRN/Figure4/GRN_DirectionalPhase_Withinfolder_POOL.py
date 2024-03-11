from GRN_directional_evolution_phas_GeneExpressionAsTrait import *

# ----------------------------------------------------------------------------------------------------------
# Global simulation parameters
n_pop = 1000
n_genes = 5
n_envs = n_genes
n_generations = 16
optima_means = np.array((7.5, 7.5)) #np.arange(0.1,5,n_generations)
mut_variance = 0.2
mutations_per_individuals=0.005
delta_t = 1
max_steps = int(3e2)
epsilon = 2  # Number of genes that are affected by environmental queues
var_e = 0.2
cor_e_e= 0.9
omega=1
# ----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    combinations=[]
    folders=()

    optima = (np.array((7.5, 12.5)),
              np.array((12.5, 7.5)))

    names = ("opt_01","opt_10") # Optimum upwards (7.5, 12.5) and to the right (12.5, 7.5), respectively

    for opt_index in range(2):
          for core in (0,,1,6,8,9,11,24,26,29,30,36,41,43,44,47, 
                      55,56,57,58,59,63,64,67,68,70,76,78,83,90,92):

            for ind in range(0,25):

                 store = (n_pop, n_genes, n_generations,
                         "evolve_plastic_" + str(core),
                         "/" + str(names[opt_index]) + "_clone_" + str(ind),
                         mut_variance, mutations_per_individuals,
                         epsilon, cor_e_e, var_e,
                         3,  # Seed
                         True,  # Plasticity
                         optima[opt_index],
                         ind,
                         omega)  # Changing environment
                 combinations.append(store)


    with Pool(6) as pool:
       pool.starmap(evolution_directional_phase, combinations)

