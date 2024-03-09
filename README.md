# DevDyn
This repository contains data and code for the results of Milocco, L. & Uller, T. (2024). Utilizing developmental dynamics for evolutionary prediction and control. PNAS. The provided codes are in Python 3.9.

Contents: 
* Folder “ReactionDiffusion” contains the code to generate data for Figure 2 of the paper. The python code gray_scott_Sensitivity.py simulates the spatially-discretized Gray-Scott model and calculates the sensitivity vectors, starting with the initial conditions provided in subfolder SIMULATE. 
* Folder “GRN” contains code and data for the gene regulatory networks results shown in Figures 3 and 4. The script GRN_development.py contains the functions to simulate development of the gene regulatory network.
  + Subfolder “Figure 3” contains the script GRN_predict_mutations.py and GRN_predict_alignment_env_vs_gen.py which were used to generate panels A and B of Figure 3, respectively.
  + Subfolder “Figure 4” contains the data and script to generate Figure 4. 
