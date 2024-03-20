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

    def __init__(self, population_size: int, chromosome_length: int, chromosome_min: int,  chromosome_max: int,  sp: float,  crossover_prob:float, mutation_rate: float, elitism_num: int):
        """
        Initialize the OneMaxGA instance.

        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Length of each chromosome (bitstring).
            mutation_rate (float): Probability of mutation for each bit.
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.elitism_num = elitism_num
        self.chromosome_min = chromosome_min
        self.chromosome_max = chromosome_max
        self.population = self.initialize_population()
        self.best_fitness_values = []
        self.average_fitness_values = []

    def create_individual(self) -> List[int]:
        """
        Create a new individual (random bitstring).

        Returns:
            List[int]: A newly created individual.
        """
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]    
    
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
        total_fitness = 0
        for chromosome in self.population:
            total_fitness += self.evaluate_fitness(chromosome)

        relative_fitness = []
        for chromosome in self.population:
            relative_fitness.append(self.evaluate_fitness(chromosome) / total_fitness)

        cumulative_probabilities = [relative_fitness[0]]
        for i in range(1, self.population_size):
            cumulative_probabilities.append(cumulative_probabilities[i - 1] + relative_fitness[i])

        return cumulative_probabilities


    def select_parents(self) -> List[List[int]]:
        """
        Select parents based on cumulative probabilities.

        Returns:
            List[List[int]]: Selected parents.
        """
        cumulative_probabilities = self.calculate_cumulative_probabilities()
        selected_parents = random.choices(self.population, cum_weights = cumulative_probabilities, k = 2)
        return selected_parents

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[List[int]]:
        """
        Perform one-point crossover between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """
        if random.uniform(0, 1) < self.crossover_prob:
            #TODO 
            crossover_point = random.randint(1, self.chromosome_length - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            return offspring1, offspring2 # type: ignore
        else:
            return parent1, parent2 # type: ignore
    

    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Apply bit flip mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.uniform(0, 1) < self.mutation_rate:
                #TODO  # Bit flip
                mutated_chromosome[i] = 1 - chromosome[i]    
        return mutated_chromosome

    def elitism(self) -> List[List[int]]:
        """
        Apply elitism to the population (keep the best two individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        sorted_population = sorted(self.population, key=self.evaluate_fitness, reverse=True)
        # TODO #return the best elitism_num
        return sorted_population[: self.elitism_num]    
        
    def decode_standard(self, chromosome: List[int]) -> Tuple[float, float]:
        """
        Decode the chromosome into real values using standard decoding.

        Args:
            chromosome (List[int]): The chromosome to decode.

        Returns:
            Tuple[float, float]: Decoded values (x1, x2).
        """
        x1 = self.chromosome_min + ((self.chromosome_max - self.chromosome_min) / (2 ** self.chromosome_length - 1)) * sum([chromosome[i] * (2 ** (self.chromosome_length - i - 1)) for i in range(self.chromosome_length // 2)])
        x2 = self.chromosome_min + ((self.chromosome_max - self.chromosome_min) / (2 ** self.chromosome_length - 1)) * sum([chromosome[i] * (2 ** (self.chromosome_length - i - 1)) for i in range(self.chromosome_length // 2, self.chromosome_length)])
        return x1, x2

    def decode_gray(self, chromosome: List[int]) -> Tuple[float, float]:
        """
        Decode the chromosome into real values using Gray decoding.

        Args:
            chromosome (List[int]): The chromosome to decode.

        Returns:
            Tuple[float, float]: Decoded values (x1, x2).
        """
        gray1 = chromosome[0]
        gray2 = chromosome[self.chromosome_length // 2]
        for i in range(1, self.chromosome_length // 2):
            gray1 = gray1 ^ chromosome[i]
            gray2 = gray2 ^ chromosome[self.chromosome_length // 2 + i]
        x1 = self.chromosome_min + ((self.chromosome_max - self.chromosome_min) / (2 ** self.chromosome_length - 1)) * gray1
        x2 = self.chromosome_min + ((self.chromosome_max - self.chromosome_min) / (2 ** self.chromosome_length - 1)) * gray2
        return x1, x2
    
    def evaluate_fitness(self, chromosome: List[int]) -> float:
        """
        Evaluate the fitness of an individual.

        Args:
            chromosome (List[int]): The chromosome to evaluate.

        Returns:
            float: Fitness value.
        """
        x1, x2 = self.decode_standard(chromosome)
        fitness = 8 - (x1 + 0.0317) ** 2 + x2 ** 2
        constraint_penalty = abs(x1 + x2 - 1)
        return fitness - constraint_penalty


    def linear_rank_selection(self):
        pop_fitness = [self.evaluate_fitness(indv) for indv in self.population]
        ranks = np.array(pop_fitness).argsort().argsort() + 1
        print("Ranks:",ranks)
        print("----------")
        pop_linear_rank_fitness = [(2-sp) + 2 * (sp - 1) * (rank-1)/(self.population_size -1) for rank in ranks]
        return pop_linear_rank_fitness

    def run(self, max_generations):
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
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
                parent1, parent2 = self.select_parents()
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
    sp=random.uniform(1,2)
    start = time.time()
    onemax_ga = OneMaxGA(population_size, chromosome_length,chromosome_min,chromosome_max,sp,crossover_prob, mutation_rate,elitism_num)
    best_solution = onemax_ga.run(max_generations)
    ga_time = time.time()-start
    print("GA Solution Time:",round(ga_time,1),'Seconds')
    print(f"Best solution: {best_solution}")
    print(f"Fitness: {onemax_ga.evaluate_fitness(best_solution)}")
    onemax_ga.plot_fitness_values()
