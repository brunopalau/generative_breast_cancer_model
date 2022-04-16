# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:04:19 2022

@author: User
"""
delay = 500

#libraries
import pygame
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng as rng
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
def generate_markov(cells):
    '''
    update cell state according to transition matrix with interaction frequencies
    '''
    #put in first cell
    total_states = cells.dimx * cells.dimy
    y,x = np.argwhere(cells.states==0)[np.random.randint(0,total_states)]
    cells.states[y,x] = np.random.randint(1,cells.number_celltype+1)
    #animate first cell if needed
    if cells.ani:
        pygame.draw.rect(cells.surface,cells.color[cells.states[y,x]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1))
        pygame.display.update()
        
    last_state = cells.states[y,x]
    
    #random walk till are states are non-zero
    while len(np.argwhere(cells.states == 0)) > 0:
        possible_moves = open_neighbors(cells,x,y)
        #if no moves possible from position x,y then search for new position
        if len(possible_moves) == 0:
            y,x,last_state = new_start(cells)
            possible_moves = open_neighbors(cells,x,y)
            #if no moves possible with new start we go back to start and check if there are still open positions
            if len(possible_moves) == 0:
                continue
            
        #choose one of the possible moves
        delta_y, delta_x = possible_moves[np.random.randint(0,len(possible_moves))]
        
        #compute new position
        new_x, new_y = x+delta_x, y+delta_y
        
        #assign new position value according to transition matrix
        random = np.random.random()
        for i,prob in enumerate(cells.transition_matrix[last_state-1]):
            if random < prob:
                new_state = i + 1
                break
        cells.states[new_y,new_x] = new_state
        
        #update current state and value
        last_state = new_state
        x,y = new_x,new_y
        
        if cells.ani:
            pygame.draw.rect(cells.surface,cells.color[cells.states[y,x]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1))
            pygame.time.delay(delay)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                else:
                    break
    if cells.ani:  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    else:
        return
   
        
def open_neighbors(cells,x,y):    
    #compute possible moves at position x,y in cells.state    
    moves = np.array([[[-1,-1],[-1,0],[-1,1]],
                 [[0,-1],[0,0],[0,1]],
                 [[1,-1],[1,0],[1,1]]])

    left_bound = x-1
    left = 0
    right_bound = x+2
    right = 3
    upper_bound = y-1
    upper = 0
    lower_bound = y+2
    lower = 3
    #check if at right or left boundary
    if x == cells.dimx-1:
        right_bound = x+1
        right -=1
    elif x == 0:
        left_bound = 0
        left +=1
    #check if at upper or lower boundary
    if y == cells.dimy-1:
        lower_bound = y+1
        lower -=1
    elif y == 0:
        upper_bound = 0
        upper +=1
    
    neig = cells.states[upper_bound:lower_bound,left_bound:right_bound]
    moves = moves[upper:lower,left:right]
    open_pos = neig == 0
    possible_moves = moves[open_pos]
    
    #no possible moves
    if not possible_moves.any():
        return []
    else:
        return possible_moves

def neighbors(cells,x,y):
    #compute possible moves at position x,y in cells.state    
    positions = np.array([[[y-1,x-1],[y-1,x],[y-1,x+1]],
              [[y,x-1],[x,y],[y,x+1]],
              [[y+1,x-1],[y+1,x],[y+1,x+1]]])
    
    left_bound = x-1
    left = 0
    right_bound = x+2
    right = 3
    upper_bound = y-1
    upper = 0
    lower_bound = y+2
    lower = 3
    #check if at right or left boundary
    if x == cells.dimx-1:
        right_bound = x+1
        right -=1
    elif x == 0:
        left_bound = 0
        left +=1
    #check if at upper or lower boundary
    if y == cells.dimy-1:
        lower_bound = y+1
        lower -=1
    elif y == 0:
        upper_bound = 0
        upper +=1
    
    neig = cells.states[upper_bound:lower_bound,left_bound:right_bound]
    positions = positions[upper:lower,left:right]
    open_pos = neig != 0
    possible_moves = positions[open_pos]
    
    #no possible moves
    if not possible_moves.any():
        return []
    else:
        return possible_moves
    
def new_start(cells):
    open_pos = np.argwhere(cells.states==0)
    new_y,new_x = open_pos[np.random.randint(0,len(open_pos))]
    neig = neighbors(cells, new_x, new_y)
    #if no neighbors around
    #todo:choose new start-point with neigbors to continue chain
    if len(neig) == 0:
        cells.states[new_y,new_x] = np.random.randint(1,cells.number_celltype+1)
        last_state = cells.states[new_y,new_x]
        x,y = new_x, new_y     
    #choose one of the neighbors randomly and continue walk
    else:
        neig_y, neig_x = neig[np.random.randint(0,len(neig))]
        last_state = cells.states[neig_y,neig_x]
        random = np.random.random()

        for i,prob in enumerate(cells.transition_matrix[last_state-1]):
            if random < prob:
                new_state = i + 1
                break
        cells.states[new_y,new_x] = new_state
        
        #update current state and value
        last_state = new_state
        x,y = new_x,new_y
        
    if cells.ani:
        pygame.time.delay(delay)
        pygame.draw.rect(cells.surface,cells.color[cells.states[y,x]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1))
        pygame.display.update
        
    print(f"new start at: {x},{y} with state {last_state}")
    return y,x,last_state



def update_mle():
    pass
    #go through every cell in matrix
    # for x, y in np.ndindex(cells.states.shape):
    #     current_state = cells.states[x,y]
    #     random = np.random.random()
    #     #decide new state depending on transition matrix
    #     for i,prob in enumerate(cells.transition_matrix[current_state-1]):
    #         if random < prob:
    #             new_state = i + 1
    #             break
    #     #set new state
    #     cells.states[x,y] = new_state
    #     #animation would better be moved to class method but this way less loops
    #     if cells.ani:
    #         pygame.draw.rect(cells.surface,cells.color[cells.states[x,y]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1))


#classes
class Board():
    
    def __init__(self,dimx,dimy,transition_matrix,number_celltype=27,cellsize=8,iterations=100):
        self.dimx = dimx
        self.dimy = dimy
        self.cellsize = cellsize
        self.number_celltype = number_celltype
        self.color = outer
        self.ani = False
        self.states = np.zeros((self.dimx,self.dimy),dtype=int)
                
        self.transition_matrix = transition_matrix
        #use diagonal matrix for testing
        # self.transition_matrix = diag_matrix
        
        #normalize transition matrix so values add to 1
        for i,row in enumerate(self.transition_matrix):
            norm_row = row/np.sum(row)
            #generate cumulative sum
            cum_sum = np.cumsum(norm_row,dtype=float)
            self.transition_matrix[i] = cum_sum


#animation function   
def animation(board,generate_func=generate_markov):
    #set update function
    board.update = generate_func

    # initialize pygame
    board.ani = True
    pygame.init()
    board.surface = pygame.display.set_mode((board.dimx * board.cellsize, board.dimy * board.cellsize))
    pygame.display.set_caption("Generative Breast Cancer Model")
    
    # #draw gid and first state
    board.surface.fill(col_grid)
    pygame.display.update()
    
    #start computation
    generate_func(board)


def generate(board,update_func=generate_markov):
    pass

def main():
    #todo: separate animation from class
    #todo: separate simulation from class
    ct_interactions = np.genfromtxt("C:/Users/User/Desktop/data/cellular_automata_bio394/ct_interactions_knn8.csv",delimiter=",",skip_header=True)
    b = Board(10,10,transition_matrix=ct_interactions,cellsize=100)
    animation(b)

if __name__ == "__main__":
    main()