import copy
import os
import numpy as np
from numpy.random import rand
from GRN_development import *


def evolution_directional_phase (n_pop,n_genes,n_generations,folder1,folder2,mut_variance,mutations_per_individual,
               epsilon,cor_e_e,var_e,seed,
               plasticity,
               optimum, indi,omega):

    delta_t = 1
    max_steps = int(3e2)

    folder=folder1+folder2
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + "/parameters.txt",
              'wt') as filename:
        filename.write("Number of genes = " + str(n_genes) + "\n")
        filename.write("Epsilon = " + str(epsilon) + "\n")
        filename.write("Mut freq = " + str(mutations_per_individual) + "\n")
        filename.write("Mut var = " + str(mut_variance) + "\n")
        filename.write("Covariance env = " + str(cor_e_e) + "\n")
        filename.write("Mut env = " + str(var_e) + "\n")
        filename.write("seed = " + str(seed) + "\n")
        filename.write("Plasticity? = " + str(plasticity) + "\n")
        filename.write("omega = " + str(omega) + "\n")
        filename.write("optimum = " + str(optimum) + "\n")

    offspring_M = []
    offspring_B = []
    offspring_Z = []
    offspring_W = []
    offspring_EE = []

    # ----------------------------------------------------------------------
    # Read from file
    with open(folder1 + "/generation_499/population.pkl", 'rb') as filename:
        [offspring_Initial,
         offspring_B,
         offspring_M,
         offspring_Z,
         offspring_W,
         offspring_E,
         _,
         _]=\
            pickle.load(filename)
    # ----------------------------------------------------------------------


    #MAKE CLONES!!!!!!!
    M=offspring_M[indi]
    B=offspring_B[indi]
    Z=offspring_Z[indi]
    W=offspring_W[indi]
    EE=offspring_E[indi]


    offspring_M = []
    offspring_B = []
    offspring_Z = []
    offspring_W = []
    # ----------------------------------------------------------------------

    for repeat in range(n_pop):
        offspring_M.append(copy.deepcopy(M))
        offspring_B.append(copy.deepcopy(B))
        offspring_Z.append(copy.deepcopy(Z))
        offspring_W.append(copy.deepcopy(W))
        offspring_EE.append(copy.deepcopy(EE))



    for generation in range(0,n_generations):
        optimum = optimum


        store_M = offspring_M
        store_B = offspring_B
        store_Z = offspring_Z
        store_W = offspring_W
        store_EE = offspring_EE
        store_Initial = offspring_Initial

        directory = folder + "/generation_" + str(generation)
        if not os.path.exists(directory):
            os.makedirs(directory)


        environment = np.repeat(0.0, n_genes)
        store_E = []
        for repeat in range(n_pop):
            store_E.append(copy.deepcopy(environment))

        store_fitness = []
        store_traits = []

        for individual in range(0, n_pop):
            [fitness, traits]= development_linear_GeneExpressionAsTrait(store_Initial[individual],
                     store_M[individual],
                     store_B[individual],
                     store_E[individual],
                     store_W[individual],
                     store_Z[individual],
                     folder,
                     generation,
                     individual,
                     optimum,
                     max_steps,
                     delta_t,
                     n_genes,
                     omega)

            store_traits.append(traits)
            store_fitness.append(fitness)

        # -----------------------------------------------------------------------------------------------------------
        # Save all genotypes
        with open(folder + "/generation_" + str(generation) + "/population.pkl", 'wb') as filename:
            pickle.dump([store_Initial, store_B, store_M, store_Z, store_W, store_E,store_fitness,store_traits], filename)
        # -----------------------------------------------------------------------------------------------------------


        relative_fitness = store_fitness / np.sum(store_fitness)

        print(folder + ", gen " + str(generation) + ", fitness =" + str(np.mean(store_fitness)))

        offspring_M = []
        offspring_B = []
        offspring_Z = []
        offspring_W = []
        offspring_Initial = []

        for ind in range(0, n_pop):
            index1=np.random.uniform()
            sum = 0
            for i in range (0,n_pop):
                sum += relative_fitness[i]
                if sum > index1:
                    index_parent1 = i
                    break

            index2=np.random.uniform()


            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            sum = 0
            i=0
            counter=0
            while i < n_pop:
                sum += relative_fitness[i]
                if sum > index2:
                    index_parent2 = i
                    if (index_parent2==index_parent1):
                        sum=0
                        i=-1
                        index2 = np.random.uniform()
                    else:
                        break
                i+=1
                counter+=1
                if counter > 1000:
                    index_parent2 = np.random.choice(n_pop)
                    break


            if (index_parent2==index_parent1):
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print ((index_parent1, index_parent2))

            # Recombination
            child_M = np.zeros((n_genes, n_genes))
            child_B = np.zeros((n_genes, n_genes))
            child_Z = np.zeros((n_genes, 2))
            child_W = np.zeros((n_genes, 2))
            child_Initial = np.zeros((n_genes))
            #M

            for loci in range(0, n_genes):
                child_Initial[loci] = store_Initial[np.random.choice((index_parent1, index_parent2))][loci]

                choose=np.random.choice((index_parent1, index_parent2))
                child_Z[loci, :] = store_Z[choose][loci,:]
                child_W[loci, :] = store_W[choose][loci,:]

                for loci2 in range(0,n_genes):
                    choose=np.random.choice((index_parent1, index_parent2))
                    child_M[loci,loci2] = store_M[choose][loci,loci2]
                    child_B[loci, loci2] = store_B[choose][loci,loci2]



            # Mutate

            number_of_mutations = np.random.poisson(mutations_per_individual, 1)

            for mut in range(int(number_of_mutations)):
                # Decide which type of mutation
                index_type_of_mutation = np.random.uniform()
                if index_type_of_mutation > 0.2:
                    i = np.random.choice(range(0, n_genes))
                    j = np.random.choice(range(0, n_genes))
                    child_B[i][j] += np.random.normal(loc=0, scale=mut_variance)
                else:
                    i = np.random.choice(range(0, n_genes))
                    j = np.random.choice(range(0, n_genes))
                    child_M[i][j] = np.absolute(child_M[i][j] - 1)


            offspring_M.append(child_M)
            offspring_B.append(child_B)
            offspring_W.append(child_W)
            offspring_Z.append(child_Z)
            offspring_Initial.append(child_Initial)
