from deap import base, algorithms
from deap import creator
from deap import tools

import algelitism

import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


LOW, UP = -5, 5
ETA = 20
LENGTH_CHROM = 2

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 150
HALL_OF_FAME_SIZE = 5

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]


toolbox = base.Toolbox()
toolbox.register("randomPoint", randomPoint, LOW, UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def func(individual):
    x, y = individual
    # f = np.sin(5 * np.cos(x) / 2) + np.cos(y)
    f = np.sin(2) * np.sqrt(x ** 2 + y ** 2) / (np.sqrt(x ** 2 + y ** 2))
    return f,


toolbox.register("evaluate", func)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)



population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

best = hof.items[0]
print(best)
plt.ioff()
plt.show()

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Покоління')
plt.ylabel('Макс/ Середня пристосованість')
plt.title('Залежність пристосування від покоління')
plt.show()


def z(x, y):
    return np.sin(2) * np.sqrt(x ** 2 + y ** 2) / (np.sqrt(x ** 2 + y ** 2) + 0.001)
print(z(*best))


# LOW = -5.0
# UP = -LOW
x = np.arange(LOW, UP, 0.05)
y = np.arange(LOW, UP, 0.05)

X, Y = np.meshgrid(x, y)
zs = np.array(z(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(best[0], best[1], z(*best), c="r", s=50)
plt.show()



