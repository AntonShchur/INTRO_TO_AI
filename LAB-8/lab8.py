import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
MAX_LENGTH = 40
MAX_HEIGHT = 60
NUMBER_OF_PLATES = 24
MAX_ITERATIONS = 1000
LEARNING_RATE = 0.001

CONNECTIONS_MATRIX = np.random.randint(0, NUMBER_OF_PLATES, size=(NUMBER_OF_PLATES, NUMBER_OF_PLATES))
CONNECTIONS_MATRIX = np.tril(CONNECTIONS_MATRIX) + np.tril(CONNECTIONS_MATRIX, -1).T
np.fill_diagonal(CONNECTIONS_MATRIX, 0)

PLATES_SHAPES = []
while sum(a * b for (a, b) in PLATES_SHAPES) < MAX_LENGTH * MAX_HEIGHT and len(PLATES_SHAPES) < NUMBER_OF_PLATES:
    a = np.random.uniform(1, int(MAX_LENGTH / 4))
    b = np.random.uniform(1, int(MAX_HEIGHT / 5))
    PLATES_SHAPES.append((a, b))


fits_best = lambda: min(CHROMOSOMES, key=lambda chromosomes: chromosomes[1])

CHROMOSOMES = []
while len(CHROMOSOMES) < NUMBER_OF_PLATES:
    CHROMOSOMES.append((list(), 0.0))
    z, _ = CHROMOSOMES[-1]
    for a, b in PLATES_SHAPES:
        bound_a = a / 2
        bound_b = b / 2
        x = np.random.uniform(bound_a, MAX_LENGTH - bound_a)
        y = np.random.uniform(bound_b, MAX_HEIGHT - bound_b)
        z.append((x, y))


def calculated_fitness(input_chromosome):
    i = 0
    length = 0
    area = 0
    while i < len(input_chromosome) - 1:
        u = i + 1
        x1, y1 = input_chromosome[i]
        a1, b1 = PLATES_SHAPES[i]
        while u < len(input_chromosome):
            x2, y2 = input_chromosome[u]
            a2, b2 = PLATES_SHAPES[u]
            distance = pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), 0.5)
            length += distance * CONNECTIONS_MATRIX[i][u]
            r1 = [x1 - a1 / 2, y1 - b1 / 2, x1 + a1 / 2, y1 + b1 / 2]
            r2 = [x2 - a2 / 2, y2 - b2 / 2, x2 + a2 / 2, y2 + b2 / 2]
            if r1[0] >= r2[2] or r1[2] <= r2[0] or r1[3] <= r2[1] or r1[1] >= r2[3]:
                area += 0
            else:
                area += (0.5 * (a2 + a1) - abs(x2 - x1)) * (0.5 * (b2 + b1) - abs(y2 - y1))
            u += 1
        i += 1
    return length, area


def crossover():
    new_generation = list()
    chroms_sorted = sorted(CHROMOSOMES, key=lambda ch: ch[1])[:int(MAX_LENGTH / 2)]
    while len(new_generation) < NUMBER_OF_PLATES:
        p1 = np.random.randint(0, len(chroms_sorted))
        p2 = np.random.randint(0, len(chroms_sorted))
        while p1 == p2:
            p2 = np.random.randint(0, len(chroms_sorted))
        a = copy.deepcopy(chroms_sorted[p1][0])
        b = copy.deepcopy(chroms_sorted[p2][0])
        bound = np.random.randint(1, len(a) - 1)
        new_chromosome_1 = a[:bound] + b[bound:]
        new_chromosome_2 = b[:bound] + a[bound:]
        for c in [new_chromosome_1, new_chromosome_2]:
            if np.random.randint(0, 100) < 20:
                gen_i = np.random.randint(0, len(c))
                gen = c[gen_i]

                mutation_x = gen[0] + np.random.uniform(-MAX_LENGTH / 100, MAX_LENGTH / 100)
                mutation_y = gen[1] + np.random.uniform(-MAX_HEIGHT / 100, MAX_HEIGHT / 100)
                if mutation_x > 0 and mutation_x < MAX_LENGTH:
                    c[gen_i] = (mutation_x, gen[1])
                if mutation_y > 0 and mutation_y < MAX_HEIGHT:
                    c[gen_i] = (mutation_x, mutation_y)
                new_generation.append((c, 0.0))


    return new_generation



def genetic_algorithm():
    global CHROMOSOMES
    iterations = []
    for i in range(MAX_ITERATIONS):
        for j in range(len(CHROMOSOMES)):
            chromosome, _ = CHROMOSOMES[j]
            length, area = calculated_fitness(chromosome)
            fitness = LEARNING_RATE * length + area
            CHROMOSOMES[j] = (chromosome, fitness)
            if area == 0:
                iterations.append((i, CHROMOSOMES[j][1]))
                return CHROMOSOMES[j], iterations
        iterations.append((i, fits_best()[1]))
        if i == MAX_ITERATIONS:
            break
        CHROMOSOMES = crossover()
    return fits_best(), iterations


result, iterations = genetic_algorithm()
centers, fitness = result
fig, ax = plt.subplots()
ax.set_xlim(0, MAX_LENGTH)
ax.set_ylim(0, MAX_HEIGHT)
ax.set_xlabel("X----------------------------------->")
ax.set_ylabel("Y----------------------------------->")
ax.set_title(f"Схема після {iterations[-1][0]} ітерацій")
i = 0

for i in range(len(PLATES_SHAPES)):
    w, h = PLATES_SHAPES[i]
    x, y = centers[i]
    xy = (x - 0.5 * w, y - 0.5 * h)
    ax.add_patch(Rectangle(xy, w, h, edgecolor='black', facecolor='blue'))
    ax.text(xy[0], xy[1], i + 1, color="w")

plt.show()
fig, ax = plt.subplots()
ax.plot(tuple(i for i, _ in iterations), tuple(f for _, f in iterations))
ax.set_xlabel("Ітерації")
ax.set_ylabel("Значення функції пристосування")
ax.set_title(f"Залежність значення пристосування від к-сті ітерацій")
plt.show()
