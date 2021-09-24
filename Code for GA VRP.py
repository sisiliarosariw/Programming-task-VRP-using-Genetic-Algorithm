# -*- coding: utf-8 -*-
"""
Created on fri Feb 26 10:51:26 2021

@author: sisil
"""

import numpy as np
import pandas as pd
from collections import Counter

# Genetic Algorithm parameter variabel setting
crossOverRate = 0.8
mutationRate = 0.1
popSize = 20      
iteration = 100
numofcar = 3
capofcar = 100
numOfDest = 15

# Generate random population of chromosome
destination = np.arange(1,16)
initialPop = np.zeros((popSize, numOfDest), dtype=int) 
for i in range(popSize):
    initialPop[i,:] = np.random.choice(destination, numOfDest, replace=False)
print('Initial chromosome:')
print(initialPop)

# Getting demand data
demand = pd.read_csv('GA-DEMAND.csv', header=0, index_col=0)
demand = demand.values

#demand each route in initial population
demandpop = np.zeros((popSize, numOfDest), dtype=int)
for i in range(popSize):
    for j in range(numOfDest):
        demandpop[i,j]= demand[initialPop[i,j]-1][0]
demandcopy = demandpop.copy()
print('demand chromosome')
print(demandpop)

#constraint three car with capacity constraint
car_populasi = np.zeros((popSize, numOfDest), dtype=int)
for i in range(popSize):
    demandcopy = demandpop.copy()
    for k in range(numofcar):
        mobil = capofcar        
        for j in range(numOfDest):  
            if demandcopy[i,j] < mobil:
                mobil = mobil - demandcopy[i,j]
                car_populasi[i,j] = k
                demandcopy[i,j] = 9999
            else:
               continue
print('chromosome mobil')
print(car_populasi)

# Getting distances data 
dist = pd.read_csv('rutejarak.csv', header=0, index_col=0)
dist = dist.values

# Calculate fitness (total distance per chromosom) for all car: depot - multiple destinations - depot 
fitness0 = np.zeros((popSize, 1), dtype=int)
for i in range(initialPop.shape[0]):
     # distance from depot to first destination
    totalDist = dist[0,initialPop[i,0]]
    for j in range(initialPop.shape[1]-1):
        if car_populasi[i,j]==0:
            start = initialPop[i,j]
            end = initialPop[i,j+1]
            totalDist = totalDist + dist[start,end]
        # distance from last destination back to depot
    totalDist = totalDist + dist[end,0]    
    fitness0[i,0] = totalDist
print('Initial fitness kendaraan 1:', fitness0[:,0])



fitness1 = np.zeros((popSize, 1), dtype=int)
for i in range(initialPop.shape[0]):
     # distance from depot to first destination
    totalDist = dist[0,initialPop[i,0]]
    for j in range(initialPop.shape[1]-1):
        if car_populasi[i,j]==1:
            start = initialPop[i,j]
            end = initialPop[i,j+1]
            totalDist = totalDist + dist[start,end]
        # distance from last destination back to depot
    totalDist = totalDist + dist[end,0]    
    fitness1[i,0] = totalDist
print('Initial fitness kendaraan 2:', fitness1[:,0])

fitness2 = np.zeros((popSize, 1), dtype=int)
for i in range(initialPop.shape[0]):
    # distance from depot to first destination
    totalDist = dist[0,initialPop[i,0]]
    for j in range(initialPop.shape[1]-1):
        if car_populasi[i,j]==2:
            start = initialPop[i,j]
            end = initialPop[i,j+1]
            totalDist = totalDist + dist[start,end]
        # distance from last destination back to depot
    totalDist = totalDist + dist[end,0]    
    fitness2[i,0] = totalDist
print('Initial fitness kendaraan 3:', fitness2[:,0])

totalfitness = []
for i in range(initialPop.shape[0]):        
    total_fitness= fitness0[i,0]+fitness1[i,0]+fitness2[i,0]
    totalfitness.append(total_fitness)
totalfitness = np.array(totalfitness)
print('Initial fitness:', totalfitness)

# This initial chromosome will be used for the first iteration   
population = initialPop.copy()

# History best iteration and best all
bestIt = np.zeros((iteration,1), dtype=int)
bestChrIt = np.zeros((iteration,numOfDest), dtype=int)
bestcarIt = np.zeros((iteration,numOfDest), dtype=int)
bestAll = np.zeros((iteration,1), dtype=int)
bestChrAll = np.zeros((iteration,numOfDest), dtype=int)
bestcarall = np.zeros((iteration,numOfDest), dtype=int)

for it in range(iteration):
    print('___________________________________________________________________')
    print('ITERASI #'+str(it))
    print('___________________________________________________________________')  
    # Setting Roulette Wheel to choose parents
    inverse = 1/totalfitness        # The goal is to find the shortest distance (minimize), then 1 / fitness 
    chance = inverse/sum(inverse)   # Calculate the proportion of the area of each chromosome
    cumulative = np.array(np.cumsum(chance)).reshape(-1,1) 
    
 # Select parents based on roulette wheel
    indexParents = []   # list to record index from chromosome selected 
    for i in range(int(population.shape[0]/2)):
        # Parent #1
        ind1 = 0
        random = np.random.rand()
        for j in range(population.shape[0]):
            if random < cumulative[j,0]:
                ind1 = j
                break
        indexParents.append(ind1)
        
        # Parent #2 
        ind2 = ind1
        while ind2 == ind1:
            random = np.random.rand()
            for j in range(population.shape[1]):
                if random < cumulative[j,0]:
                    ind2 = j
                    break
        indexParents.append(ind2)
    parents = population[indexParents,:].copy()
    print('Parents selected:')
    print(parents)

    
    # Cross over
    print('\nCross over:')
    offspring = np.zeros((parents.shape[0], parents.shape[1]), dtype=int)
    for i in range(0,parents.shape[0],2):
        r = round(np.random.rand(), 4) 
        print(i, r, crossOverRate)
        if r <= crossOverRate:
            child1, parent1 = parents[i,:].copy(), parents[i,:].copy()
            child2, parent2 = parents[i+1,:].copy(), parents[i+1,:].copy()
            # print(parent1, parent2)
            low = np.random.randint(0,parents.shape[0]-1)
            if low == parents.shape[0] - 1:
                up = parents.shape[1]
            else:
                up = np.random.randint(low+1,parents.shape[0])
            # print(low, up)
            child1[low:up] = parent2[low:up]
            child2[low:up] = parent1[low:up]
            # print(child1, child2)
            
            childCheck = []    
            for child in [child1, child2]:
                frequency = [item for item, count in Counter(child).items() if count > 1]
                # unique, counts = np.unique(child, return_counts=True)
                # frequencies = np.asarray((unique, counts)).T
                # frequency = frequencies[frequencies[:,1] > 1,0]
                if len(frequency) > 0:
                    missing = destination[~np.isin(destination, child)]
                    for j in range(len(frequency)):
                        child[np.where(child==frequency[j])[0][-1]] = missing[j]
                childCheck.append(child)
            # print(childCheck[0], childCheck[1])
            offspring[i,:] = childCheck[0]
            offspring[i+1,:] = childCheck[1]
        else:
            offspring[i,:] = parents[i,:].copy()
            offspring[i+1,:] = parents[i+1,:].copy()
            # print(offspring[i,:], offspring[i+1,:])
            
    #Getting demand in chromosome
    demandpop = np.zeros((popSize, numOfDest), dtype=int)
    for i in range(popSize):
        for j in range(numOfDest):
            demandpop[i,j]= demand[offspring[i,j]-1][0]
    print('demand chromosome')
    print(demandpop)

    #Getting car in chromosome
    carpop = np.zeros((popSize, numOfDest), dtype=int)
    for i in range(popSize):
        demandcopy = demandpop.copy()
        for k in range(numofcar):
            mobil = capofcar        
            for j in range(numOfDest):  
                if demandcopy[i,j] < mobil:
                    mobil = mobil - demandcopy[i,j]
                    carpop[i,j] = k
                    demandcopy[i,j] = 9999
                else:
                    continue
    demandcopy = demandpop.copy()
    print('chromosome mobil')
    print(carpop)
            
    # Mutation
    print('\nMutation:')
    for i in range(offspring.shape[0]): 
        p = round(np.random.rand(), 4) 
        print(i, p, mutationRate)
        if p <= mutationRate:
            # print(offspring[i,:])
            bits = np.random.choice(np.arange(offspring.shape[1]), 2, replace=False)
            temp = offspring[i,bits[0]]
            offspring[i,bits[0]] = offspring[i,bits[1]]
            offspring[i,bits[1]] = temp
            # print(bits, offspring[i,:])
    print('\noffspring (after crossover and mutation):')
    print(offspring)
    
    #Getting demand in chromosome
    demandpop = np.zeros((popSize, numOfDest), dtype=int)
    for i in range(popSize):
        for j in range(numOfDest):
            demandpop[i,j]= demand[offspring[i,j]-1][0]
    # print('demand chromosome')
    # print(demandpop)

    #Getting car in chromosome
    carpop = np.zeros((popSize, numOfDest), dtype=int)
    for i in range(popSize):
        demandcopy = demandpop.copy()
        for k in range(numofcar):
            mobil = capofcar        
            for j in range(numOfDest):  
                if demandcopy[i,j] < mobil:
                    mobil = mobil - demandcopy[i,j]
                    carpop[i,j] = k
                    demandcopy[i,j] = 9999
                else:
                    continue
    demandcopy = demandpop.copy()
    print('chromosome mobil')
    print(carpop)
    
    # Calculate offspring fitness (per chromoseome): depot - multiple destinations - depot
    fitness0 = np.zeros((popSize, 1), dtype=int)
    for i in range(offspring.shape[0]):
        # distance from depot to first destination
        totalDist = dist[0,offspring[i,0]]
        for j in range(offspring.shape[1]-1):
            if carpop[i,j]==0:
                start = offspring[i,j]
                end = offspring[i,j+1]
                totalDist = totalDist + dist[start,end]
        # distance from last destination back to depot
        totalDist = totalDist + dist[end,0]    
        fitness0[i,0] = totalDist
    print('Intitial fitness kendaraan 1:', fitness0[:,0])

    # Calculate fitness (total distance per chromoseome): depot - multiple destinations - depot 
    fitness1 = np.zeros((popSize, 1), dtype=int)
    for i in range(offspring.shape[0]):
        # distance from depot to first destination
        totalDist = dist[0,offspring[i,0]]
        for j in range(offspring.shape[1]-1):
            if carpop[i,j]==1:
                start = offspring[i,j]
                end = offspring[i,j+1]
                totalDist = totalDist + dist[start,end]
        # distance from last destination back to depot
        totalDist = totalDist + dist[end,0]    
        fitness1[i,0] = totalDist
    print('Intitial fitness kendaraan 2:', fitness1[:,0])


    # Calculate fitness (total distance per chromoseome): depot - multiple destinations - depot 
    fitness2 = np.zeros((popSize, 1), dtype=int)
    for i in range(offspring.shape[0]):
        # distance from depot to first destination
        totalDist = dist[0,offspring[i,0]]
        for j in range(offspring.shape[1]-1):
            if carpop[i,j]==2:
                start = offspring[i,j]
                end = offspring[i,j+1]
                totalDist = totalDist + dist[start,end]
        # distance from last destination back to depot
        totalDist = totalDist + dist[end,0]    
        fitness2[i,0] = totalDist
    print('Intitial fitness kendaraan 3:', fitness2[:,0])

    totalfitness = []
    for i in range(offspring.shape[0]):        
        total_fitness= fitness0[i,0]+fitness1[i,0]+fitness2[i,0]
        totalfitness.append(total_fitness)
    totalfitness = np.array(totalfitness)
    print('Intitial fitness:', totalfitness)

    best = totalfitness.min()
    bestIndex = np.where(totalfitness == best)[0][0]
    bestChromosome = offspring[bestIndex,:].copy()
    bestcarpop = carpop[bestIndex,:].copy()
    # Recording best from each iteration 
    bestIt[it,:] = best
    bestChrIt[it,:] = bestChromosome
    bestcarIt[it,:] = bestcarpop
    print('Best fitness:', best, '\nChromosom at index no.:', bestIndex, 
          '\nBest chromosome:', bestChromosome)  
    print('Best car:', bestcarpop)  
    
    # Recording best of the best (from all iteration) 
    if it == 0: 
        bestAll[it,:] = best
        bestChrAll[it,:] = bestChromosome
        bestcarall[it,:] = bestcarpop
    else:
        if best < bestAll[it-1,:]:
            bestAll[it,:] = best
            bestChrAll[it,:] = bestChromosome
            bestcarall[it,:] = bestcarpop
        else:
            bestAll[it,:] = bestAll[it-1,:]
            bestChrAll[it,:] = bestChrAll[it-1,:].copy()
            bestcarall[it,:] = bestcarall[it-1,:].copy()

    # This offspring chromosome will be used for the next iteration  
    population = offspring.copy() 
    print('------------------------------\n')  

print('Best Solution from all iteration:')  
# pd.DataFrame(bestIt, columns=['best fitness iteration']).plot.bar()
pd.DataFrame(bestAll, columns=['best fitness']).plot()
print('Best fitness from', iteration, 'iteration:', bestAll.min(), 
      '\nFirst time achieved at iteration:', np.where(bestAll[:,0] == bestAll.min())[0][0],  
       '\nBest chromosome from', iteration) 
print (bestChrAll[np.where(bestAll[:,0] == bestAll.min())[0][0],:])
print('Best car: ',  bestcarall[np.where(bestAll[:,0] == bestAll.min())[0][0],:])

import time

# starting time
start = time.time()

# program body starts
for i in range(10):
    print(i)

# sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

# program body ends

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")