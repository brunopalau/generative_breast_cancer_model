# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:04:19 2022

@author: User
"""
import pygame
import numpy as np
import seaborn as sns

outer = []
#create color
for i in range(1,28):
    inner = []
    for i in range(3):
        inner.append(np.random.randint(1,257))
    outer.append(tuple(inner))

col_background = (10, 10, 40)
col_grid = (30, 30, 60)

meta_clusters = {1:"B cell",2:"T & B cells",3:"T cell",4:"Macrophage",5:"T cell",
                 6:"Macrophage",7:"Endothelial",8:"Vimentin hi",9:"Small circular",10:"Small elongated",
                 11:"Fibronectin hi",12:"Large elongated",13:"SMA hi Vimentin hi",14:"Hypoxic",15:"Apoptotic",
                 16:"Proliferative",17:"p53+ EGFR+",18:"Basal CK",19:"CK7+ CK hi Ecadh hi",20:"CK7+",
                 21:"Epithelial low",22:"CK lo HR lo",23:"CK+ HR hi",24:"CK+ HR+",25:"CK+ HR lo",
                 26:"CK lo HR hi p53+",27:"Myoepithelial"}

class Board():
    
    def __init__(self,dimx,dimy,number_celltype=27,cellsize=8,ani=False):
        self.dimx = dimx
        self.dimy = dimy
        self.cellsize = cellsize
        self.number_celltype = number_celltype
        
        self.states = np.random.randint(1,self.number_celltype,size=(self.dimx,self.dimy))
        #self.transition_matrix = np.random.random((number_celltype,number_celltype))
        self.ani = ani
        if self.ani:
            self.color = outer #sns.color_palette("hls",number_celltype)
            self.animation()
        
    def update(self):
        for x, y in np.ndindex(self.states.shape):
            self.states[x,y] = np.random.randint(1,self.number_celltype)
            if self.ani:
                pygame.draw.rect(self.surface,self.color[self.states[x,y]-1],(self.cellsize*x,self.cellsize*y,self.cellsize-1, self.cellsize-1))
    
    def animation(self):
        pygame.init()
        self.surface = pygame.display.set_mode((self.dimx * self.cellsize, self.dimy * self.cellsize))
        pygame.display.set_caption("John Conway's Game of Life")
        
        self.surface.fill(col_grid)
        for x, y in np.ndindex(self.states.shape):
            pygame.draw.rect(self.surface,self.color[self.states[x,y]],(self.cellsize*x,self.cellsize*y,self.cellsize-1, self.cellsize-1))
        pygame.display.update()

    
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
    
            self.surface.fill(col_grid)
            self.update()
            pygame.display.update()


def main():
    b = Board(50,50,ani=True)

if __name__ == "__main__":
    main()