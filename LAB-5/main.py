import numpy as np
import imageio
import glob
import os

IMAGE_SHAPE = (140, 140)


def predict(input_image, input_weights):

    pixel_sum = np.dot(input_weights, input_image.T)
    max_idx = np.argmax(pixel_sum)
    answer = np.array([-1, -1, -1, -1])
    answer[max_idx] = 1
    return answer


def load_images(path):
    image_names = os.listdir(f'{path}/')
    data = []
    for path in glob.glob(f"{path}/*.png"):
        image_sample = imageio.imread(path, format="png")[:, :, 0].reshape(140 ** 2).astype("int")
        for i in range(len(image_sample)):
            if image_sample[i] == 255:
                image_sample[i] = -1
            else:
                image_sample[i] = 1
        data.append(image_sample)
    data = np.array(data)

    targets_dict = {}
    n_samples = data.shape[0]
    targets = np.full((n_samples, n_samples), -1)
    for i in range(n_samples):
        targets[i, i] = 1
        targets_dict[','.join(str(x) for x in targets[i])] = image_names[i][0]

    return data, targets, targets_dict


train_images, train_targets, train_targets_dict = load_images("./images")


def fit(train_x, train_y):
    weights = np.zeros(train_x.shape)
    while True:
        for idx, image in enumerate(train_x):
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    weights[i, j] = weights[i, j] + image[j] * train_y[idx, i]

        predicts = []
        for image in train_x:
            predicts.append(predict(image, weights))
        predicts = np.array(predicts)

        if predicts.all() == train_y.all():
            break
    return weights


trained_weights = fit(train_images, train_targets)
test_images, test_targets, test_dict = load_images("./test")


for image, test_targets in zip(test_images, test_targets):
    prediction = predict(image, trained_weights)
    prediction = ','.join(str(x) for x in prediction)
    print(prediction)
    test_targets = ','.join(str(x) for x in test_targets)
    print(f"{test_dict[test_targets]} розпізнало як {train_targets_dict[prediction]}")


print("Погані картинки")
test_wrong_images, test_wrong_targets, test_wrong_dict = load_images("./wrong_samples")

for image, test_targets_wrong in zip(test_wrong_images,test_wrong_targets):
    prediction = predict(image, trained_weights)
    prediction = ','.join(str(x) for x in prediction)
    test_targets = ','.join(str(x) for x in test_targets_wrong)
    print(f"{test_wrong_dict[test_targets]} розпізнало як {train_targets_dict[prediction]}")


