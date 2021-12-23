import pandas as pd
import numpy as np


def z_func(x, y):
    return np.sin(5 * np.cos(x) / 2) + np.cos(y)


x = np.arange(start=0, stop=1, step=0.01, dtype=float)
y = np.arange(start=0, stop=1, step=0.01, dtype=float)
z = z_func(x, y)


writer = pd.ExcelWriter("data.xlsx")
df = pd.DataFrame({"X":x, "Y":y,"Z":z})

df.to_excel(writer, "data.xlsx")
writer.save()