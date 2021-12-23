import numpy as np
import skfuzzy
import matplotlib.pyplot as plt
import pandas as pd

GAUSS = 1
TRIANGLE = 2
TRAPEZOID = 3


# def y_func(x):
#     return np.sin(x) + np.cos(x)*np.sin(np.cos(x))
#
#
# def z_func(x, y):
#     return np.sin(5 * np.cos(x) / 2) + np.cos(y)

def y_func(x):
    return np.sin(x) + np.cos(x / 2)


def z_func(x, y):
    return np.sin(2) * np.sqrt(x ** 2 + y ** 2) / (np.sqrt(x ** 2 + y ** 2) + 0.001)



x = np.arange(0, 1.01, 0.001)
z = z_func(x, y_func(x))
plt.plot(x,z)
plt.show()

STEP = 0.0001
Z_COUNT = 9
X_COUNT = Y_COUNT = 6
z_min = 1.15
z_max = 1.7
x_min = 0
x_max = 1
y_min = 0
y_max = 1


def create_mfs(func_amount, f_min, f_max):
    gauss = []
    tri = []
    tra = []
    input_range = np.arange(f_min, f_max, 0.001)
    for i in range(func_amount):
        mid = f_min + i * (f_max - f_min) / (func_amount - 1)
        sigma = input_range.std()/6
        gauss.append(skfuzzy.gaussmf(input_range, mid, sigma))
        tri.append(skfuzzy.trimf(input_range, [mid - 2.5*sigma, mid, mid + 2.5*sigma]))
        tra.append(skfuzzy.trapmf(input_range, [mid - 2.5*sigma, mid - 0.5*sigma, mid + 0.5*sigma, mid + 2.5*sigma]))
    return input_range, gauss, tri, tra


def draw_mf(input_range, gauss, tri, tra):
    for plot in gauss:
        plt.plot(input_range, plot)
    plt.show()
    for plot in tri:
        plt.plot(input_range, plot)
    plt.show()
    for plot in tra:
        plt.plot(input_range, plot)
    plt.show()


z_range, z_gauss, z_tri, z_tra = create_mfs(Z_COUNT, z_min, z_max)
x_range, x_gauss, x_tri, x_tra = create_mfs(X_COUNT, x_min, x_max)
y_range, y_gauss, y_tri, y_tra = create_mfs(Y_COUNT, y_min, y_max)

draw_mf(z_range, z_gauss, z_tri, z_tra)
draw_mf(x_range, x_gauss, x_tri, x_tra)
draw_mf(y_range, y_gauss, y_tri, y_tra)


def get_mz(value):
    values = []
    for i in np.arange(z_min, z_max, (z_max - z_min) / (Z_COUNT-1)):
        values.append(skfuzzy.gaussmf(value, i, z_range.std() / 6))
    return np.argmax(values)


def get_mf(value, mf_type, mf_min, mf_max, mf_amount):
    values = []
    step = (mf_max - mf_min) / (mf_amount-1)
    input_range = np.arange(mf_min, mf_max, 0.001)
    sigma = input_range.std() / mf_amount
    if mf_type == GAUSS:
        for i in np.arange(mf_min, mf_max, step):
            values.append(skfuzzy.gaussmf(value, i, sigma))
        return np.argmax(values)
    elif mf_type == TRIANGLE:
        for i in np.arange(mf_min, mf_max, step):
            values.append(skfuzzy.trimf(np.array([value]), [i - 2.5 * sigma, i, i + 2.5 * sigma]))
        return np.argmax(values)
    elif mf_type == TRAPEZOID:
        for i in np.arange(mf_min, mf_max, step):
            values.append(skfuzzy.trapmf(np.array([value]), [i - 2.5*sigma, i - 0.5*sigma, i + 0.5*sigma, i + 2.5*sigma]))
        return np.argmax(values)


def get_my(value):
    values = []
    for i in np.arange(y_min, y_max, (y_max - y_min) / (Y_COUNT-1)):
        values.append(skfuzzy.gaussmf(value, i, y_range.std() / 6))
    return np.argmax(values)


df = pd.DataFrame(dtype=str)
input_x = np.arange(x_min, x_max + 0.01, 0.2)
y = y_func(input_x)
z = np.zeros((6, 6))

for i in range(6):
    df[f"my{i}"] = np.array([]*6)
    for j in range(6):
        z[i, j] = z_func(input_x[i], y[j])

df["index"] = np.array([f"mx{i}" for i in range(6)])
df.set_index('index', inplace=True)
for i in range(6):
    for j in range(6):
        df.at[f'mx{i}', f'my{j}'] = int(get_mf(z[i, j], GAUSS, z_min, z_max, 9) + 1)
print(df)
print(z)
def test():
    random_x = np.array(sorted(np.random.uniform(0, 1, (100,))))
    # x_range = np.arange(0.3, 1, 0.001)
    z_list = []
    for x in random_x:
        mx = get_mf(x, GAUSS, x_min, x_max, 6)
        my = get_mf(x, GAUSS,x_min, x_max, 6)
        z_list.append(z[mx, my])
    z_list = np.array(z_list)
    fig, ax = plt.subplots()
    ax.plot(random_x, z_list)
    ax.plot(x_range, z_func(x_range, y_func(x_range)))
    plt.show()
    error = np.mean(np.abs(z_func(random_x, y_func(random_x)) - z_list) / (z_func(random_x, y_func(random_x))+1)) * 100
    print(f'{round(error,2)}%')


test()