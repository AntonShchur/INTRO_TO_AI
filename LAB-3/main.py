import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fcmeans import FCM


def main():
    df = pd.read_csv("Iris.csv", delimiter=",")

    train_data = df[["SepalLengthCm", "SepalWidthCm"]]
    data = train_data.astype(dtype=float).to_numpy()

    model = FCM(n_clusters=3)
    model.fit(data)
    predict = model.predict(data)

    centroids = model.centers
    goals = model.get_goals()
    train_steps = len(goals)
    _, axes = plt.subplots(1, figsize=(8, 5))
    axes.plot([x for x in range(train_steps)], goals)
    axes.set_xlabel("К-сть ітерацій")
    axes.set_ylabel("Значення функції")
    plt.show()
    _, axes = plt.subplots(1, figsize=(8, 5))

    axes.scatter(df["SepalLengthCm"].to_numpy(), df["SepalWidthCm"].to_numpy(), c=predict)
    axes.scatter(centroids[:, 0], centroids[:, 1], marker="+", c="black", s=500)
    axes.set_xlabel("SepalLengthCm")
    axes.set_ylabel("SepalWidthCm")
    axes.set_title("fuzzy C means")
    plt.show()

if __name__ == "__main__":
    main()