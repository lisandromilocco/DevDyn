import copy
import os
import numpy as np
import pylab as pl
from numpy.random import rand
from numpy.random import randint
from GRN_development import *
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
# ----------------------------------------------------------------------------------------------------------
# Global simulation parameters
n_pop = 10
n_genes = 5
n_envs = n_genes
n_generations = 16
folder = 'evolve_plastic'
perturbation_variance=.5
optimum = np.array((5.0, 5.0)) #np.arange(0.1,5,n_generations)
delta_t = 1
max_steps = int(3e2)
epsilon = 2  # Number of genes that are affected by environmental queues


# global_parameters = (n_pop, n_loci, n_param, n_generations, folder, optimum, mut_size, mut_freq)
# ----------------------------------------------------------------------------------------------------------
pl.rcParams['font.family'] = 'serif'
cmfont = fm.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
pl.rcParams['font.serif'] = cmfont.get_name()
pl.rcParams['mathtext.fontset'] = 'cm'
pl.rcParams['axes.unicode_minus'] = False


names = ("opt_10", "opt_01")

def draw_fitness(opt_index,row,ax):

        colorlist= ()
        store_store_store_store_fitness = []
        store_store_store_store_fitness_01 = []
        store_store_store_store_fitness_10 = []

        for core in (0,1,6,8,9,11,24,26,29,30,36,41,43,44,47,55,56,57,58,59,63,64,67,68,70,76,78,83,90,92):#range(0,100):#
            store_store_store_fitness = []


            for ind in range(25):#range(0,25):

                if row==0:
                    folder="evolve_"+"plastic"+"_" + str(core)+"/" + str(names[opt_index]) + "_clone_" + str(ind)

                elif row==1:
                    folder = "evolve_" + "plastic" + "_" + str(core) + "/" + str(names[opt_index]) + "_Accelerated_clone_" + str(ind)
                else:
                    print("error***")

                if not os.path.isfile(folder + "/generation_"+str(n_generations-1)+"/population.pkl"):
                   print(folder + "/generation_"+str(n_generations)+"/population.pkl does not exist")
                   continue


                store_store_fitness=[]
                for generation in range(0,n_generations):

                    f=folder + "/generation_"+str(generation)+"/population.pkl"
                    if not os.path.isfile(f):
                        print(f + "does not exist")
                        continue

                    # ----------------------------------------------------------------------
                    # Read from file
                    with open(f, 'rb') as filename:
                        [_,
                         _,
                         _,
                         _,
                         _,
                         _,
                         store_fitness,
                         _] = \
                            pickle.load(filename)  # Read from file

                    store_store_fitness.append(store_fitness)

                ###!!!!!!!!!!!!!!!!!!!!1
                if len(store_store_fitness) == 0:
                    continue
                ###!!!!!!!!!!!!!!!!!!!!!!!

                colormap = pl.get_cmap('magma')
                color1 = colormap(0)
                color2 = colormap(0.7)

                dummy=np.average(store_store_fitness, axis=1)
                compress=(dummy-dummy[0])*1000#/dummy[0]

                store_store_store_fitness.append(compress)
                if (core <= 49):
                    color=color1
                else:
                    color=color2

                #ax_array[row,opt_index].plot(range(0, n_generations), np.average(store_store_fitness, axis=1), c=color, alpha=0.05)

            ###!!!!!!!!!!!!!!!!!!!!1
            if len(store_store_store_fitness) == 0:
                   continue
            ###!!!!!!!!!!!!!!!!!!!!!!!

            ax.plot(range(0, n_generations), np.average(store_store_store_fitness, axis=0),
                     c=color, alpha=.25) #,label=str(core) )

            if core<= 49:
                store_store_store_store_fitness_01.append(np.average(store_store_store_fitness, axis=0))
            else:
                store_store_store_store_fitness_10.append(np.average(store_store_store_fitness, axis=0))
        #pl.legend(loc="upper left")

#!!!!!!!!!!!!!!!!!!!!


        mu=np.average(store_store_store_store_fitness_01, axis=0)

        ax.plot(range(0, n_generations), mu,
                c=color1, alpha=1)  # ,label=str(core) )


        mu = np.average(store_store_store_store_fitness_10, axis = 0)

        ax.plot(range(0, n_generations), mu,
                c=color2, alpha=1)  # ,label=str(core) )






				

# set up figure
fig = pl.figure(figsize=(10, 3))
gs = matplotlib.gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1,1])

# Panel A

store_store_deviations=[]
store_store_colors=[]
store_store_traits=[]

core_list=[6,57]

colormap = pl.get_cmap('magma')
color1 = colormap(0)
color2 = colormap(0.7)

axxx = fig.add_subplot(gs[0, 0])

for index in range(2):
    core=core_list[index]

    folder="evolve_plastic_"+str(core)
    if not os.path.isfile(folder + "/generation_499/population.pkl"):
        print(folder + "/generation_499/population.pkl does not exist")
        continue

    # ----------------------------------------------------------------------
    # Read from file
    with open(folder + "/generation_499/population.pkl", 'rb') as filename:
        [offspring_Initial,
         store_B,
         store_M,
         store_Z,
         store_W,
         store_E,
         _,
         _] = \
            pickle.load(filename)  # Read from file

    store_Initial = [np.random.uniform(.1, .1, size=n_genes) for _ in range(n_pop)]
    # ----------------------------------------------------------------------

    combinations = []
    cov = np.identity(epsilon)
    cov[cov == 0.0] = cov_e_e
    cov = cov * perturbation_variance

    for individual in range(0, n_pop):
        for environment in range(0, n_envs):
            if environment==0:
                E = np.repeat(0.0, n_genes)
            else:
                #------------------------------------------
                E = np.repeat(0.0, n_genes)
                i = np.random.choice(range(0, epsilon))
                E[i]=np.random.normal(0,perturbation_variance)
                if (environment==1):
                    E[i] = .5
                else:
                    E[i] = -.5

                #E[i] = np.random.uniform(-1, 1,1)
                #------------------------------------------

            store = (store_Initial[individual],
                 store_M[individual],
                 store_B[individual],
                 E,
                 store_W[individual],
                 store_Z[individual],
                 folder,
                 9999999,
                 individual,
                 environment,
                 max_steps,
                 delta_t,
                 n_genes)

            combinations.append(store)

    if not os.path.exists(folder+"/perturbation"):
        os.makedirs(folder+"/perturbation")


    st = time.time()
    with Pool(7) as pool:
        pool.starmap(development_perturbation, combinations)
    et = time.time()
    print("time taken parallel =", et - st)
    # -----------------------------------------------------------------------------------------------------------

    store_traits = []
    #for individual in range(0, n_pop):
    colors = []
    colormap = pl.get_cmap('magma')
    color = colormap(0.4)
    for i in range(n_pop):
        colors.append(colormap(np.random.uniform(0,1,1)))
        #colors.append('#%06X' % randint(0, 0xFFFFFF))

    store_deviations=[]
    store_colors=[]

    if (core <= 49):
        color = color1
    else:
        color = color2
#

    for individual in range(0,n_pop):
        store_ind=[]
        store_Es=[]
        for environment in range(0, n_envs):
            with open(folder + "/perturbation/individual_" + str(individual) +"_environment"+ str(environment) +".pkl", "rb") as filename:
                PPP = pickle.load(filename)
            store_ind.append(PPP[0])
            store_Es.append(np.sum(PPP[1]))

        X = [item[0] for item in store_ind]
        Y = [item[1] for item in store_ind]

        if (X[0]<6 or Y[0]<6):
            continue

        axxx.scatter(X[0], Y[0],
                   #c=colors[individual],
                   marker=".",
                   color=color)



        for environment in range(0, n_envs):

            prop = dict(arrowstyle="->,head_width=0.2,head_length=0.8",
                        shrinkA=0, shrinkB=0,
                        #color=colors[individual],
                        color=color,
                        alpha=.5)

            if(abs(Y[environment]-Y[0])>0.01):
                axxx.annotate("", xy=(X[environment], Y[environment]),
                         xytext=(X[0], Y[0]), arrowprops=prop)

            #
            # pl.plot([ X[0], X[environment] ],
            #         [ Y[0], Y[environment] ], c=colors[individual],alpha=1)

            store_deviations.append(store_ind[environment]-store_ind[0])
            store_store_deviations.append(store_ind[environment]-store_ind[0])
            if store_Es[environment]>=0:
                store_colors.append("red")
                store_store_colors.append("red")
            else:
                store_colors.append("blue")
                store_store_colors.append("blue")

        axxx.set_ylim([6,9])
        axxx.set_xlim([6,9])


    axxx.set_xlabel(r'$x_{3}$', fontsize=12)
    axxx.set_ylabel(r'$x_{4}$', fontsize=12)

    axxx.set_xticks([6,7,8,9])
    axxx.set_yticks([6, 7, 8, 9])

# Panel B and C


ax2 = fig.add_subplot(gs[0, 1])
draw_fitness(0,0,ax2)

ax3 = fig.add_subplot(gs[0, 2])
draw_fitness(0,1,ax3)

ax2.set_xlabel("Generations")
ax2.set_ylabel("Fitness")
ax2.set_ylim([-0.05, 3])

ax3.set_xlabel("Generations")
ax3.set_ylabel("Fitness")
ax3.set_ylim([-0.05, 3])


pl.savefig("analyses/"+"plastic"+"_figure_MERGED.pdf", format="pdf", bbox_inches="tight")

pl.close()
print("done")