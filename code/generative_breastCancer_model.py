# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:04:19 2022

@author: User
"""
delay = 50

#libraries
import pygame
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.random import default_rng as rng
import seaborn as sns
from scipy.ndimage import convolve as conv

#global variables
##create color list and background colors
R = np.linspace(0,255,27, dtype=int)
G = np.linspace(0,255,27, dtype=int)
B = np.linspace(0,255,27, dtype=int)

np.random.seed(4)
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
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
    else:
        return

def generate_best_start(cells):
    '''
    update cell state according to transition matrix with interaction frequencies,
    using the whole spatial information of the positionj with the most neighbors
    '''
    
    #put in first cell
    total_states = cells.dimx * cells.dimy
    y,x = np.argwhere(cells.states==0)[np.random.randint(0,total_states)]
    cells.states[y,x] = np.random.randint(1,cells.number_celltype+1)
    #animate first cell if needed
    if cells.ani:
        pygame.draw.rect(cells.surface,cells.color[cells.states[y,x]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1))
        pygame.display.update()
            
    #random walk till all states are non-zero
    while len(np.argwhere(cells.states == 0)) > 0:
        
        y,x = best_start(cells)
        #assign new position value according to transition matrix
        #take all spatial neighbors into account
        neig = neighbors(cells, x, y)

        #create probablity vector for this cell
        count_neig = 0
        prob_v = np.zeros_like(cells.transition_matrix[0,:])
        for s in neig:
            if s == 0:
                continue
            else:
                prob_v += cells.transition_matrix[s-1]
                count_neig += 1
        #normalize probablity vector
        prob_v = prob_v/count_neig
                
        #choose state
        random = np.random.random()
        for i,prob in enumerate(prob_v):
            if random < prob:
                new_state = i + 1
                break
        cells.states[y,x] = new_state
        
        
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
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
    else:
        return
    
def generate_conv(cells):
    '''
    update cell state according to transition matrix with interaction frequencies,
    using the whole spatial information
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
    
    #random walk till all states are non-zero
    while len(np.argwhere(cells.states == 0)) > 0:
        possible_moves = open_neighbors(cells,x,y)
        #if no moves possible from position x,y then search for new position
        if len(possible_moves) == 0:
            new_y,new_x = best_start(cells)
            print(f"new start at: {new_y},{new_x}")
            #if no moves possible with best_start we are finished
            if new_y == -1:
                break
        else:    
            #choose one of the possible moves
            delta_y, delta_x = possible_moves[np.random.randint(0,len(possible_moves))]
            
            #compute new position
            new_x, new_y = x+delta_x, y+delta_y
            
        #assign new position value according to transition matrix
        #take all spatial neighbors into account
        neig = neighbors(cells, new_x, new_y)

        #create probablity vector for this cell
        count_neig = 0
        prob_v = np.zeros_like(cells.transition_matrix[0,:])
        for s in neig:
            if s == 0:
                continue
            else:
                prob_v += cells.transition_matrix[s-1]
                count_neig += 1
        #normalize probablity vector
        prob_v = prob_v/count_neig
                
        #choose state
        random = np.random.random()
        for i,prob in enumerate(prob_v):
            if random < prob:
                new_state = i + 1
                break
        cells.states[new_y,new_x] = new_state
        
        #update current state and value
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
        while True:
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

def best_start(cells):

    kernel = [[1,1,1],
              [1,0,1],
              [1,1,1]]
    defined_positions = np.array(cells.states!=0,dtype=int)
    neig_count = conv(defined_positions,kernel,mode="constant",cval=0)
    #only consider unused spaces
    neig_count[cells.states!=0] = -1
    #max number of neigbors
    maxi = np.max(neig_count)
    #if -1 then no open positions left
    if maxi == -1:
        return -1,-1
    #consider only positions with the most amount of information given
    open_pos = np.argwhere(neig_count==maxi)
    #choose one of the neighbors randomly and continue walk
    new_y,new_x = open_pos[np.random.randint(0,len(open_pos))]
   
    return new_y,new_x
    
def new_start(cells):
    open_pos = np.argwhere(cells.states==0)
    #todo:choose new start-point with neigbors to continue chain
    new_y,new_x = open_pos[np.random.randint(0,len(open_pos))]
    neig = open_neighbors(cells, new_x, new_y)
    #if no neighbors around
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

def neighbors(cells,x,y):
    #compute state of neighbors at position x,y in cells.state    

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
    states = neig.flatten()
    return states


def generate_neighborhood(cells):
    
    #initiate all states to a random cell type
    cells.states = np.random.randint(1,cells.number_celltype+1,(cells.dimy,cells.dimx))
    
    if cells.ani:
        for y,row in enumerate(cells.states):
            for x,state in enumerate(row):
                pygame.draw.rect(cells.surface,cells.color[cells.states[y,x]-1],(cells.cellsize*x,cells.cellsize*y,cells.cellsize-1, cells.cellsize-1)) 
        pygame.display.update()
    
    #todo: take a look at convolution
    #go though all cells and compute neighbors celltype probablity vector
    converging = True
    iterations = 0
    while converging:
        counter = 0
        #creat a x*y*celltypes matrix
        probs = np.zeros((cells.dimx,cells.dimy,cells.number_celltype))

        #add probablity vectors
        for y,row in enumerate(cells.states):
            for x,state in enumerate(row):
                left = max(x-1,0)
                right = min(cells.dimx+1,x+2)
                upper = max(y-1,0)
                lower = min(cells.dimy+1,y+2)
                  
                #create neighborhood filter
                cond = np.full(np.shape(cells.states),False)
                cond[upper:lower,left:right] = True
                cond[y,x] = False
                
                #choose prbability vector according to cell state in x,y
                v = cells.interactions[state-1]
                
                #add probablity vector to neighboring cells
                probs[cond,:] += v
    
            
        #update cells.state with new probablity matrix
        changes = False
        for y,row in enumerate(cells.states):
            for x,state in enumerate(row):
                prob_vector = probs[y,x,:]
                #normalize probablity vector and compute cumulative sum
                prob_vector = prob_vector/np.sum(prob_vector)
                prob_vector = np.cumsum(prob_vector,dtype=float)
                
                # (deterministic) #choose most likely cell type
                # new_state = np.argmax(prob_vector) + 1
                
                #choose state probabilistically
                random = np.random.random()
                for i,prob in enumerate(prob_vector):
                    if random < prob:
                        new_state = i + 1
                        break
                #if changed
                if new_start != state:
                    #assign new cell state
                    counter += 1
                    cells.states[y,x] = new_state
                    changes = True
                                
                #animation would better be moved to class method but this way less loops
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
        print(f"changes made: {counter}")
        #if no more changes are happening
        if not changes:
            print("converged")
            converging = False
        #no convergence after x iterations
        if iterations == 100:
            print(f"no convergence after {iterations} iterations")
            converging = False
        iterations+=1
        
    if cells.ani:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
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
        
        self.interactions = transition_matrix
        self.transition_matrix = np.ones_like(self.interactions)
        
        #normalize transition matrix so values add to 1
        for i,row in enumerate(self.interactions):
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


def generate(board,generate_func=generate_markov,plot=False):
    generate_func(board)
    if plot == True:
        fig, ax = plt.subplots()
        cmap = cm.get_cmap(name='Reds')
        ax.imshow(board.states, cmap = "tab20b", origin = "lower")




def main():
    ct_interactions = np.genfromtxt("C:/Users/User/Desktop/data/cellular_automata_bio394/ct_interactions_knn8.csv",delimiter=",",skip_header=True)
    dimx, dimy = 60,60
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
    
    e = Board(dimx,dimy,transition_matrix=ct_interactions,cellsize=cellsize)
    # animation(e,generate_func=generate_best_start)
    generate(e,plot=True,generate_func=generate_best_start)

if __name__ == "__main__":
    main()