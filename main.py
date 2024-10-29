import pandas as pd
from Perceptron import *
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def load_data():
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df = df.drop(columns = ["island", "year", "sex"])
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['species'])
    X = df.drop(columns = "species")
    X = X.to_numpy()
    y = df["species"]
    y = y.to_numpy()
    return (X, y)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def main():
    X, y = load_data()

    train_data = X[:int(len(X)*.8)]
    train_labels = y[:int(len(X)*.8)]

    test_data = X[int(len(X)*.8):]
    test_labels = y[int(len(X)*.8):]

    classifier = Perceptron(learning_rate = 0.1, n_iters = 20)
    classifier.fit(train_data, train_labels)
    print("Computed weights: ", classifier.weights)
    predictions = classifier.predict(test_data)
    print("Perceptron classification accuracy", accuracy(test_labels, predictions))

if __name__ == "__main__":
    main()
