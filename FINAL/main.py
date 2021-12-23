import tensorflow.keras
import pygad.kerasga
import numpy as np
import pygad
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse




input_layer = tensorflow.keras.layers.Input(2)
dense_layer1 = tensorflow.keras.layers.Dense(6, activation="softmax")(input_layer)
dense_layer2 = tensorflow.keras.layers.Dense(8, activation="softmax")(dense_layer1)
dense_layer3 = tensorflow.keras.layers.Dense(8, activation="softmax")(dense_layer2)
dense_layer4 = tensorflow.keras.layers.Dense(6, activation="softmax")(dense_layer3)
dense_layer5 = tensorflow.keras.layers.Dense(6, activation="softmax")(dense_layer4)
output_layer = tensorflow.keras.layers.Dense(1, activation="linear")(dense_layer5)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)



def fitness_func(solution, _):
    global data_inputs, data_outputs, keras_ga, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,  weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)

    prediction = model.predict(data_inputs)
    return -mse(data_outputs, prediction)



def z(x, y):
    return np.cos(x) * x + np.sin(y)


X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
Z = z(X, Y)

data_inputs = np.vstack((X, Y)).T
data_outputs = z(X, Y).T


num_generations = 1000
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations, num_parents_mating=8,
                       initial_population=initial_population, fitness_func=fitness_func, mutation_type="random",
                       mutation_num_genes=10,
                       parent_selection_type="rank")

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Найкраще значення функції = {solution_fitness}")
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
metric = mse(data_outputs, predictions)
print("Помилка: ", metric)

_, axes = plt.subplots()
axes.plot(X, predictions, label="simulated")
axes.plot(X, data_outputs, label="target")
axes.set_xlabel("X")
axes.set_ylabel("Z")
axes.set_title("Моделювання функції двох змінних")
plt.legend()
plt.show()