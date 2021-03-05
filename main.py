import numpy as np
import cv2
import tqdm
from collections import defaultdict
import itertools
import random
import pandas as pd
from parametrs import *
from life_classes import *

def create_map(population,screen_size):
    cell_map = np.full(screen_size + [3],0)
    for cell in population.population:
        if population.population_map[cell.y,cell.x] != -1:
            
            cell_map[cell.y,cell.x] = cell.color
            

    return cell_map

a = Population()
i = 0
data_population = pd.Series([])

try:

    while sum(a.life_cell):
        draw_cell_map = create_map(a,screen_size)
        draw_medium = a.medium
        draw = draw_medium * 0.5 + draw_cell_map * 1 + 0
        draw = draw.transpose((2,0,1))
        img1 = cv2.merge((draw[2],draw[1],draw[0]))# Use opencv to merge as b,g,r
        img1 = cv2.resize(img1,(800,800),interpolation = cv2.INTER_AREA)
        cv2.imwrite('out.png',img1)
        
        img = cv2.imread('out.png')
        cv2.imshow('cell',img)
        cv2.waitKey(1)
        a.step_life()
        data_population = data_population.append(a.population[a.life_cell == False])
        i += 1
        if i%10 == 0:
            print('{}\nPopulation size: {},\nEpoch: {}\n{}'.format(30*'-',len(a.population[a.life_cell]),a.time,30*'-'))
            a.population = a.population[a.life_cell]
            a.life_cell = a.life_cell[a.life_cell]

finally:    
    cv2.destroyAllWindows()

    data_population = data_population.append(a.population[a.life_cell == True])

    print('Create csv file....')
    data_population = data_population.unique()

    life_data = pd.DataFrame({
        'Genome':[],
        'time_create':[],
        'mother':[],
        'age':[]
    })
    for i in data_population:
        life_data.loc[i.idx,:] = [' '.join(i.genome),i.time_create,i.mother,i.age]
        
        
    life_data = life_data.sort_index()
    life_data.iloc[:,1:4] = life_data.iloc[:,1:4].astype(int)
    life_data.to_csv('life_data.csv')
    print('Done! Bye!')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#video=cv2.VideoWriter('video1.avi', fourcc, 3,(1000,1000))
#for i in tqdm.notebook.tqdm(range(310)):
#    img = cv2.imread('image\out_{}.png'.format(i,cv2.IMREAD_UNCHANGED))
#    img = cv2.resize(img,(1000,1000),interpolation = cv2.INTER_AREA)
#    video.write(img)
#cv2.destroyAllWindows()
#video.release()