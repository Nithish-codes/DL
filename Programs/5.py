import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Flatten, Dense)

lfw_data = fetch_lfw_people(min_faces_per_person = 100, resize=0.4)

x = lfw_data.images
y = lfw_data.target
target_names = lfw_data.target_names
n_classes = len(target_names)

x = x.reshape(
    -1,
    x.shape[1],
    x.shape[2],
    1
) / 255.0

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

model = Sequential([
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape = (x.shape[1], x.shape[2], 1)
    ),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(
        128,
        activation="relu"
    ),
    Dense(
        n_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy']
)

model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))

test_loss, test_accuracy = model.evaluate(x_test,y_test)
print(test_accuracy)

def predict_and_display(index):
    sample = x_train[index].reshape(1,x.shape[1], x.shape[2],1)

    prediction = model.predict(sample)
    predicted_index = np.argmax(prediction)

    predicted_label = target_names[predicted_index]
    actual_label = target_names[y_test[index]]

    plt.imshow(
        x_test[index].reshape(
            x.shape[1],
            x.shape[2]
        ),
        cmap='gray'
    )

    plt.title(
        f"Predicted: {predicted_label}\n"
        f"Actual: {actual_label}"
    )

    plt.axis('off')
    plt.show()

predict_and_display(5)