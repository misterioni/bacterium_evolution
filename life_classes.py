import numpy as np
import cv2
import tqdm
from collections import defaultdict
import itertools
import random
import pandas as pd
from parametrs import *

class Cell():
    
    def __init__(self,time,genome = ''):
        genes_list = np.array(['Up', 'Down', 'Left', 'Right', 'Foto', 'Chemo', 'Eat'])
        if len(genome) == 0:
            genome = np.array([random.choice(genes_list) for _ in np.arange(len_genome)])
        unique, counts = np.unique(genome,return_counts = True)
        gene_count = defaultdict(int,zip(unique,counts))
        
        color = np.array([
            (gene_count['Eat']*255)/(sum([gene_count['Eat'],gene_count['Foto'],gene_count['Chemo']])) if gene_count['Eat'] !=0 else 0,
            (gene_count['Foto']*255)/(sum([gene_count['Eat'],gene_count['Foto'],gene_count['Chemo']])) if gene_count['Eat'] !=0 else 0,
            (gene_count['Chemo']*255)/(sum([gene_count['Eat'],gene_count['Foto'],gene_count['Chemo']])) if gene_count['Eat'] !=0 else 0
        ])
        if sum(color) == 0:
            color = np.array([255,255,255])
        
        self.screen_size = np.array(screen_size)
        
        self.age = 0
        self.organic = start_organic
        
        self.gene_count = gene_count
        self.color = np.around(color)
        self.genome = genome
        self.genes_list = genes_list
        
        self.idx = None
        self.x = None
        self.y = None
        
        self.time_create = time
        self.mother = -1
        
    def mutation(self, p_type_mutation = [0.95,0.03,0.01,0.01]):
        type_mutation = np.random.choice(['Not_Mutation','SNP','Insertion','Deletion'], p = p_type_mutation)
        if type_mutation == 'Not_Mutation':
            pass
        if type_mutation == 'SNP':
            self.genome[random.randint(0,len(self.genome))-1] = np.random.choice(self.genes_list)
        if type_mutation == 'Insertion':
            insertion = [random.randint(0,len(self.genome)-1) for _ in range(2)]
            self.genome = np.append(self.genome,self.genome[min(insertion):max(insertion)])
        if type_mutation == 'Deletion':
            deletion = [random.randint(0,len(self.genome)-1) for _ in range(2)]
            self.genome = np.delete(self.genome,np.arange(min(deletion) , max(deletion)))
        
        unique, counts = np.unique(self.genome,return_counts = True)
        gene_count = defaultdict(int,zip(unique,counts))
        self.gene_count = gene_count    
        self.color = np.array([
            (gene_count['Eat']*255)/(sum([gene_count['Eat'],gene_count['Foto'],gene_count['Chemo']])) if gene_count['Eat'] !=0 else 0,
            (gene_count['Foto']*255)/(sum([gene_count['Eat'],gene_count['Foto'],gene_count['Chemo']])) if gene_count['Eat'] !=0 else 0,
            (gene_count['Chemo']*255)/(sum([gene_count['Eat'],gene_count['Foto'],gene_count['Chemo']])) if gene_count['Eat'] !=0 else 0
        ])
        if sum(self.color) == 0:
            self.color = np.array([255,255,255])
        self.color = np.around(self.color)
        
    def action(self,population,population_map,medium,life_cell,dead_list, time):
        self.age += 1
        if self.idx not in dead_list:
            if self.organic <= dead_level or self.age > dead_age:
                life_cell[self.idx] = False
                population_map[self.y,self.x] = empty
                dead_list.add(self.idx)
            elif self.organic < life_level:

                gene = random.choice(self.genome)

                if gene == 'Right':
                    if self.x < self.screen_size[1] - 1:
                        if population_map[self.y, self.x + 1] == empty:
                            population_map[self.y,self.x], population_map[self.y, self.x+1] = population_map[self.y, self.x+1],population_map[self.y, self.x]
                            self.x += 1
                            self.organic -= move_score

                elif gene == 'Left':
                    if self.x > 0:
                        if population_map[self.y, self.x - 1] == empty:
                            population_map[self.y,self.x], population_map[self.y, self.x-1] = population_map[self.y, self.x-1],population_map[self.y, self.x]
                            self.x -= 1
                            self.organic -= move_score

                elif gene == 'Up':
                    if self.y > 0:
                        if population_map[self.y - 1, self.x] == empty:
                            population_map[self.y,self.x] , population_map[self.y - 1, self.x] = population_map[self.y-1,self.x] , population_map[self.y, self.x]
                            self.y -= 1
                            self.organic -= move_score

                elif gene == 'Down':
                    if self.y < self.screen_size[0] - 1:
                        if population_map[self.y + 1, self.x] == empty:
                            population_map[self.y,self.x] , population_map[self.y + 1, self.x] = population_map[self.y+1,self.x] , population_map[self.y, self.x]
                            self.y += 1
                            self.organic -= move_score


                elif gene == 'Eat':

                    neighbors = population_map[max(0,self.y - 1):min(screen_size[0] - 1, self.y + 2),
                                       max(0,self.x - 1):min(screen_size[0] - 1, self.x + 2)]

                    kill_list = list(neighbors[(neighbors != empty) & (neighbors != self.idx)])
                    if len(kill_list) != 0:
                        random.shuffle(kill_list)

                        for kill_id in kill_list:
                            if kill_id not in dead_list:

                                self.organic += organic_eat * population[kill_id].organic

                                life_cell[kill_id] = False
                                population_map[self.y,self.x] = empty
                                self.y = population[kill_id].y
                                self.x = population[kill_id].x

                                population_map[self.y,self.x] = self.idx

                                dead_list.add(kill_id)
                                break


                elif gene == 'Foto':
                    self.organic += medium[self.y,self.x,1]/100 * organic_foto
                elif gene == 'Chemo':
                    self.organic += medium[self.y,self.x,2]/100 * organic_salt 

            elif (self.x != 0 and self.x != screen_size[1] - 1)  and (self.y != 0 and self.y != screen_size[0] - 1) :

                neighbors = population_map[max(0,self.y - 1):min(screen_size[0] - 1, self.y + 2),
                                           max(0,self.x - 1):min(screen_size[0] - 1, self.x + 2)]
                if len(neighbors[neighbors != empty ]) < 9:
                    new_cell = Cell(genome = self.genome, time = time)
                    new_cell.mutation(p_type_mutation = p_mutation)
                    new_cell.organic = self.organic/2
                    new_cell.idx = population.last_valid_index() + 1
                    new_cell.mother = self.idx
                    self.organic = self.organic/2
                    coord = list(itertools.product(range(self.y - 1,self.y + 2),range(self.x - 1,self.x + 2)))
                    random.shuffle(coord)
                    for i,j in coord:
                        if (population_map[i,j] == empty) and ((i,j) != (self.y,self.x)):
                            population[population.last_valid_index() + 1] = new_cell
                            life_cell[population.last_valid_index()] = True
                            population_map[i,j] = population.last_valid_index()
                            new_cell.y = i
                            new_cell.x = j
                            break

                            
class Population():

    def __init__(self):

        self.time = 0
        self.len_genome = len_genome
        population = np.array([])
        population_map = np.full(screen_size,empty)
        life_cell = []
        for idx,(x,y) in enumerate(random.sample(list(itertools.product(range(screen_size[0]),range(screen_size[1]))),k = len_population)):
            cell = Cell(time = self.time)
            cell.x = x
            cell.y = y
            cell.idx = idx
            population_map[y,x] = idx
            population = np.append(population,cell)
            life_cell.append(True)
        self.population = pd.Series(population)
        self.population_map = population_map
        self.life_cell = pd.Series(life_cell)
        
        medium = np.zeros((screen_size[0],screen_size[1],3))
        medium[:,:,0] = medium_organic
        medium[0,:,1] = medium_light
        medium[0,:,2] = medium_salt
        for line in range(1,medium.shape[0]):
            medium[line,:,1] = medium[line-1,:,1] * gradient_light
            medium[line,:,2] = medium[line-1,:,2] * gradient_salt
        medium[:,:,2] = medium[:,:,2][::-1]
        medium = np.round_(medium)
        self.medium = medium
        

    
    def step_life(self):
        dead_list = set()
        self.population[self.life_cell].apply(lambda x: x.action(self.population,self.population_map,self.medium,self.life_cell,dead_list,self.time))  
        self.time += 1
  
        
