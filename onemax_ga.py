from matplotlib import pyplot as plt
from genetic_algorithm import GeneticAlgorithm
import random
import time
import math
import numpy as np
from typing import List, Tuple

class OneMaxGA(GeneticAlgorithm):
    """
    Genetic algorithm for solving the One-Max problem.
    Inherits from the GeneticAlgorithm abstract base class.
    """

    def __init__(self, population_size: int, number_Variable: int, R_min: List[int], R_max: List[int],  chromosome_length: int, chromosome_min: int,  chromosome_max: int,  sp: float,  crossover_prob:float, mutation_rate: float, tournament_size: int, elitism_num: int):
        """
        Initialize the OneMaxGA instance.

        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Length of each chromosome (bitstring).
            mutation_rate (float): Probability of mutation for each bit.
        """
        self.R_min = R_min
        self.R_max = R_max
        self.number_Variable = number_Variable
        self.tournament_size=tournament_size
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.elitism_num = elitism_num
        self.chromosome_min = chromosome_min
        self.chromosome_max = chromosome_max
        self.sp=sp
        self.population = self.initialize_population()
        self.best_fitness_values = []
        self.average_fitness_values = []

    def create_individual(self) -> List[int]:
        """
        Create a new individual (random bitstring).

        Returns:
            List[int]: A newly created individual.
        """
        digits = []
        for j in range(self.chromosome_length):
            if j < self.chromosome_length/2:
                r = random.SystemRandom().uniform(self.R_min[0], self.R_max[0])
                digits.append(r)
            else:
                r = random.SystemRandom().uniform(self.R_min[1], self.R_max[1])
                digits.append(r)
        return digits
    
    def initialize_population(self) -> List[List[int]]:
        """
        Initialize the population with random bitstrings.

        Returns:
            List[List[int]]: Initial population.
        """
        return [self.create_individual() for _ in range(self.population_size)]

    def calculate_cumulative_probabilities(self) -> List[float]:
        """
        Calculate cumulative probabilities for each individual.

        Returns:
            List[float]: Cumulative probabilities.
        """
        X1_arr = []
        X2_arr = []
        for i in range(self.population_size):
            chromo = self.population[i]
            for j in range(len(chromo)):
                if j < len(chromo)/2:
                    X1_arr.append(chromo[j])
                else:
                    X2_arr.append(chromo[j])
        fitness_arr = []
        sum_fitness = 0
        for i in range(self.population_size):
            fitness = 8 - ((X1_arr[i] + 0.0317)**2) + (X2_arr[i]**2)
            fitness_arr.append(fitness)
            sum_fitness = sum_fitness + fitness
        return fitness_arr
    
    def evaluate_fitness(self, chromosome: List[int]) -> float:
        """
        Evaluate the fitness of an individual.

        Args:
            chromosome (List[int]): The chromosome to evaluate.

        Returns:
            float: Fitness value.
        """
        X1_arr = []
        X2_arr = []
        for i in range(self.population_size):
            chromosome = self.population[i]
            for j in range(self.chromosome_length):
                if j < self.chromosome_length/2:
                    X1_arr.append(chromosome[j])
                else:
                    X2_arr.append(chromosome[j])
        fitness_arr = []
        sum_fitness = 0
        for i in range(self.population_size):
            fitness = 8 - ((X1_arr[i] + 0.0317)**2) + (X2_arr[i]**2)
            fitness_arr.append(fitness)
            sum_fitness = sum_fitness + fitness
        return fitness_arr

    def select_parents(self) -> List[List[int]]:
        """
        Select parents based on cumulative probabilities.

        Returns:
            List[List[int]]: Selected parents.
        """
        cumulative_probabilities = self.calculate_cumulative_probabilities()
        selected_parents = random.choices(self.population, cum_weights = cumulative_probabilities, k = 2)
        return selected_parents
    
    def calc_rank(self) -> tuple[list, list[int]]:
        fitness_arr = self.calculate_cumulative_probabilities()
        sort_fitness = []
        sort_rank_list = []
        unsort_rank_list = [0]*self.population_size
        for i in range(len(fitness_arr)):
            sort_fitness.append(fitness_arr[i])
            sort_rank_list.append(i+1)
        sort_fitness.sort()
        for i in range(len(fitness_arr)):
            for j in range(len(fitness_arr)):
                if fitness_arr[j]== sort_fitness[i]:
                    unsort_rank_list[j] = sort_rank_list[i]
                else:
                    continue
        self.best_fitness_values.append(sort_fitness[len(sort_fitness)-1])
        return fitness_arr,unsort_rank_list
    
    def linear_Rank(self):
        (fitness_arr,unsort_rank_list) = self.calc_rank()
        sp = self.sp
        length = self.population_size
        rank_fitness = []
        cumu_list = []
        prop_list = []
        total_rank = 0
        sum = 0
        for i in range(self.population_size):
            rankfit = (2-sp)+((2*(sp-1))*((unsort_rank_list[i]-1)/(length-1)))
            rank_fitness.append(rankfit)
            total_rank = total_rank + rankfit
        for i in rank_fitness:
            prop = i / total_rank
            sum = sum + prop
            cumu_list.append(sum)
            prop_list.append(prop)
        return cumu_list,fitness_arr
    
    def tournament_selection(self):
        (cumu_dist_arr,fitness_arr) = self.linear_Rank()
        contenders = []
        fit_contenders = []
        for j in range(self.tournament_size):
            r = random.uniform(0,1)
            for i in range(self.population_size):
                if i == 0:
                    if r < cumu_dist_arr[i]:
                        select = self.population[i]
                        contenders.append(select)
                    else:
                        continue
                else:
                    if cumu_dist_arr[i-1] < r and r < cumu_dist_arr[i]:
                        select = self.population[i]
                        contenders.append(select)
                    else:
                        continue
        for i in range(self.tournament_size):
            for j in range(self.population_size):
                if contenders[i] == self.population[j]:
                    fit_contenders.append(fitness_arr[j])
                else:
                    continue
        for i in range(self.tournament_size):
            if i == 0:
                best = fit_contenders[i]
                select_best = contenders[i]
            else:
                if best < fit_contenders[i]:
                    best = fit_contenders[i]
                    select_best = contenders[i]
        return select_best
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[List[int]]:
        """
        Perform one-point crossover between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """
        alpha = np.random.random()
        crossover_1 = []
        crossover_2 = []
        r = np.random.random()
        if r < 0.6:
            for i in range(2):
                for j in range(2):
                    if i == 0:
                        o1 = alpha*parent1[j] + (1-alpha)*parent2[j]
                        crossover_1.append(o1)
                    else:
                        o2 = alpha*parent2[j] + (1-alpha)*parent1[j]
                        crossover_2.append(o2)
        else:
            crossover_1 = parent1
            crossover_2 = parent2
        return crossover_1,crossover_2
    
    def mutate(self, crossover_1: List[int], crossover_2: List[int]) -> List[int]:
        """
        Apply bit flip mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        sigma = 0.5
        alpha = 0.05
        mean = 0
        mutation_1 =[]
        mutation_2 =[]
        for i in range(len(crossover_1)):
            r = np.random.random()
            if r < alpha:
                m1 = crossover_1[i] + random.gauss(mean, sigma)
                mutation_1.append(m1)
            else:
                mutation_1 = crossover_1
        for i in range(len(crossover_2)):
            r = np.random.random()
            if r < alpha:
                m2 = crossover_2[i] + random.gauss(mean, sigma)
                mutation_2.append(m2)
            else:
                mutation_2 = crossover_2
        return mutation_1,mutation_2

    def elitism(self) -> List[List[int]]:
        """
        Apply elitism to the population (keep the best two individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        (fitness_arr,unsort_rank_list) = self.calc_rank()
        best_fitness_chromo =[]
        for i in range(self.population_size):
            if unsort_rank_list[i] == self.population_size:
                best_fitness_chromo = self.population[i]
                index = i
            else:
                continue
        self.population.pop(index)
        fitness_arr.pop(index)
        unsort_rank_list.pop(index)
        return best_fitness_chromo

    def run(self, max_generations):
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1= self.tournament_selection()
                parent2= self.tournament_selection()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1,offspring2)
                new_population.extend([offspring1, offspring2])

            new_population = new_population[0:self.population_size-self.elitism_num] # make sure the new_population is the same size of original population - the best individuals we will append next
            best_individuals = self.elitism()
            new_population.extend(best_individuals)
            self.population = new_population
            best_fitness = max([self.evaluate_fitness(indv) for indv in self.population])
            average_fitness = sum([self.evaluate_fitness(indv) for indv in self.population]) / self.population_size
            self.best_fitness_values.append(best_fitness)
            self.average_fitness_values.append(average_fitness)

        best_solution = max(self.population, key=self.evaluate_fitness)
        return best_solution
    
    def runWithoutElitism(self, max_generations):
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.tournament_selection()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            self.population = new_population

            # Record best and average fitness values
            best_fitness = max([self.evaluate_fitness(indv) for indv in self.population])
            average_fitness = sum([self.evaluate_fitness(indv) for indv in self.population]) / self.population_size
            self.best_fitness_values.append(best_fitness)
            self.average_fitness_values.append(average_fitness)
        best_solution = max(self.population, key=self.evaluate_fitness)
        return best_solution
    
    def plot_fitness_values(self):
        generations = range(1, len(self.best_fitness_values) + 1)
        plt.plot(generations, self.best_fitness_values, label='Best Fitness')
        plt.plot(generations, self.average_fitness_values, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Fitness Values Over Generations')
        plt.legend()
        plt.show()
if __name__ == "__main__":
    population_size = 50
    chromosome_length = 20
    crossover_prob = 0.3
    mutation_rate = 0.01
    elitism_num = 2
    max_generations = 10  
    chromosome_min=0
    chromosome_max=31
    tournament_size=2
    sp=random.uniform(1,2)
    start = time.time()
    onemax_ga = OneMaxGA(population_size, 2, [-2,-29], [29,2], chromosome_length,chromosome_min,chromosome_max,sp,crossover_prob, mutation_rate,tournament_size,elitism_num)
    best_solution = onemax_ga.run(max_generations)
    ga_time = time.time()-start
    print("GA Solution Time:",round(ga_time,1),'Seconds')
    print(f"Best solution: {best_solution}")
    onemax_ga.plot_fitness_values()
