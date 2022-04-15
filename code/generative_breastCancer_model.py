# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:04:19 2022

@author: User
"""

#libraries
import pygame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#global variables
##create color list and background colors
outer = []
for i in range(1,28):
    inner = []
    for i in range(3):
        inner.append(np.random.randint(0,256))
    outer.append(tuple(inner))
col_background = (10, 10, 40)
col_grid = (30, 30, 60)

##metacluster dictionary
meta_clusters = {1:"B cell",2:"T & B cells",3:"T cell",4:"Macrophage",5:"T cell",
                 6:"Macrophage",7:"Endothelial",8:"Vimentin hi",9:"Small circular",10:"Small elongated",
                 11:"Fibronectin hi",12:"Large elongated",13:"SMA hi Vimentin hi",14:"Hypoxic",15:"Apoptotic",
                 16:"Proliferative",17:"p53+ EGFR+",18:"Basal CK",19:"CK7+ CK hi Ecadh hi",20:"CK7+",
                 21:"Epithelial low",22:"CK lo HR lo",23:"CK+ HR hi",24:"CK+ HR+",25:"CK+ HR lo",
                 26:"CK lo HR hi p53+",27:"Myoepithelial"}

##create diagonal matrix for testing
v = np.ones(27)
diag_matrix = np.diag(v)

#update functions
def update_markov(cells):
    '''
    update cell state according to transition matrix with interaction frequencies
    '''
    #go through every cell in matrix
    for x, y in np.ndindex(cells.states.shape):
        current_state = cells.states[x,y]
        random = np.random.random()
        #decide new state depending on transition matrix
        for i,prob in enumerate(cells.transition_matrix[current_state-1]):
            if random < prob:
                new_state = i + 1
                break
        #set new state
        cells.states[x,y] = new_state
        #animation would better be moved to class method but this way less loops
        if cells.ani:
            pygame.draw.rect(cells.surface,cells.color[cells.states[x,y]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1))

def update_mle():
    pass

#classes
class Board():
    
    def __init__(self,dimx,dimy,transition_matrix,number_celltype=27,cellsize=8,update_func=update_markov,iterations=100,ani=False):
        self.dimx = dimx
        self.dimy = dimy
        self.cellsize = cellsize
        self.number_celltype = number_celltype
        self.states = np.random.randint(1,self.number_celltype+1,size=(self.dimx,self.dimy))
        self.update = update_func
        
        self.transition_matrix = transition_matrix
        #use diagonal matrix for testing
        # self.transition_matrix = diag_matrix
        
        #normalize transition matrix so values add to 1
        for i,row in enumerate(self.transition_matrix):
            norm_row = row/np.sum(row)
            #generate cumulative sum
            cum_sum = np.cumsum(norm_row,dtype=float)
            self.transition_matrix[i] = cum_sum
        
        self.ani = ani
        if self.ani:
            self.color = outer #sns.color_palette("hls",number_celltype)
            self.animation()
            
        #todo: implement without animation
        else:
            #plot final state
            for i in iterations:
                self.update(self)
            plt.plot()
        
    def animation(self):
        # initialize pygame
        pygame.init()
        self.surface = pygame.display.set_mode((self.dimx * self.cellsize, self.dimy * self.cellsize))
        pygame.display.set_caption("Generative Breast Cancer Model")
        
        # #draw gid and first state
        self.surface.fill(col_grid)
        for x, y in np.ndindex(self.states.shape):
            pygame.draw.rect(self.surface,self.color[self.states[x,y]-1],(self.cellsize*x,self.cellsize*y,self.cellsize-1, self.cellsize-1))
        pygame.display.update()

        #continue looping till quit() and update states each run
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
    
            self.surface.fill(col_grid)
            self.update(self)
            pygame.display.update()


def main():
    #todo: separate animation from class
    #todo: separate simulation from class
    ct_interactions = np.genfromtxt("C:/Users/User/Desktop/data/cellular_automata_bio394/ct_interactions_knn8.csv",delimiter=",",skip_header=True)
    b = Board(50,50,transition_matrix=ct_interactions,ani=True)

if __name__ == "__main__":
    main()