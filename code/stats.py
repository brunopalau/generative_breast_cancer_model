# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:39:13 2022

@author: User
"""
import generative_breastCancer_model as gm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.random import default_rng as rng
import seaborn as sns
from scipy.ndimage import convolve as conv


def interactions(cells_list,plot=False):
    for cells in cells_list:
        #create interaction dictionary
        interaction = np.zeros_like(cells.transition_matrix)
        for y,row in enumerate(cells.states):
            for x,current_state in enumerate(row):
                neig = gm.neighbors(cells, x, y)
                for s in neig:
                    interaction[current_state-1,s-1] += 1
    c1 = np.sum(interaction,axis=1)
    c2 = np.sum(interaction,axis=0)
    relative_interaction = (interaction.T / c1).T
    relative_interaction = (relative_interaction/c2)
    if plot == True:
        fig, ax = plt.subplots(ncols=3)
        cmap = cm.get_cmap(name='Reds')
        ax[0].imshow(interaction, origin = "upper")
        ax[1].imshow(relative_interaction, origin = "upper")
        ax[2].imshow(cells.interactions, origin = "upper")

    return interaction
            
    
    
    
    
    
def main():
    ct_interactions = np.genfromtxt("C:/Users/User/Desktop/data/cellular_automata_bio394/ct_interactions_knn8.csv",delimiter=",",skip_header=True)
    total_freq = np.genfromtxt("C:/Users/User/Desktop/data/cellular_automata_bio394/total_freq.csv",delimiter=",")

    dimx, dimy = 70,70
    cellsize=1000/dimx
    
    #use random walk and markov for computation
    # a = Board(dimx,dimy,transition_matrix=ct_interactions,cellsize=cellsize)
    # animation(a)
    # b = Board(dimx,dimy,transition_matrix=ct_interactions,cellsize=cellsize)
    # generate(b,plot=True)
    
    #use neighborhood to compute state
    # c = Board(dimx,dimy,transition_matrix=ct_interactions,cellsize=cellsize)
    # # generate(c,generate_func=generate_neighborhood,plot=True)
    # animation(c,generate_func=generate_neighborhood)

    #use 8 spatial neighbors to determine state
    # d = Board(dimx,dimy,transition_matrix=ct_interactions,cellsize=cellsize)
    # animation(d,generate_func=generate_conv)
    # generate(d,plot=True,generate_func=generate_conv)
    list_cells = []
    for i in range(100):
        e = gm.Board(dimx,dimy,transition_matrix=ct_interactions,total_freq = total_freq,cellsize=cellsize)
        # gm.animation(e,generate_func=gm.generate_best_start)
        if i == 99:
            gm.generate(e,plot=True,generate_func=gm.generate_markov)
        else:
            gm.generate(e,plot=False,generate_func=gm.generate_markov)
            
        list_cells.append(e)
        
    inter_markov = interactions(list_cells,plot=True)
    
    list_cells = []
    for i in range(100):
        e = gm.Board(dimx,dimy,transition_matrix=ct_interactions,total_freq = total_freq,cellsize=cellsize)
        # gm.animation(e,generate_func=gm.generate_best_start)
        if i == 99:
            gm.generate(e,plot=True,generate_func=gm.generate_best_start)
        else:
            gm.generate(e,plot=False,generate_func=gm.generate_best_start)
            
        list_cells.append(e)
        
    inter_best_start = interactions(list_cells,plot=True)



    list_cells = []
    for i in range(100):
        e = gm.Board(dimx,dimy,transition_matrix=ct_interactions,total_freq = total_freq,cellsize=cellsize)
        # gm.animation(e,generate_func=gm.generate_best_start)
        if i == 99:
            gm.generate(e,plot=True,generate_func=gm.generate_conv)
        else:
            gm.generate(e,plot=False,generate_func=gm.generate_conv)
            
        list_cells.append(e)
        
    inter_conv = interactions(list_cells,plot=True)
    
    list_cells = []
    for i in range(100):
        e = gm.Board(dimx,dimy,transition_matrix=ct_interactions,total_freq = total_freq,cellsize=cellsize)
        # gm.animation(e,generate_func=gm.generate_best_start)
        if i == 99:
            gm.generate(e,plot=True,generate_func=gm.generate_neighborhood)
        else:
            gm.generate(e,plot=False,generate_func=gm.generate_neighborhood)
            
        list_cells.append(e)
        
    inter_neighborhood = interactions(list_cells,plot=True)

    list_cells = []
    for i in range(100):
        e = gm.Board(dimx,dimy,transition_matrix=ct_interactions,total_freq = total_freq,cellsize=cellsize)
        # gm.animation(e,generate_func=gm.generate_best_start)
        if i == 99:
            gm.generate(e,plot=True,generate_func=gm.mcmc)
        else:
            gm.generate(e,plot=False,generate_func=gm.mcmc)
            
        list_cells.append(e)
        
    inter_mcmc = interactions(list_cells,plot=True)
    
    
    
    
if __name__ == "__main__":
    main()
# abstand zu anderne
# no specific positions
# graph building

# total frequency
# solely, and not solely (isolated)
# average distance between cells (isolated/clustered)
# dominant celltype neighbor?

    
    
# how much clustering, total frequency per cell
# who are the neighbors