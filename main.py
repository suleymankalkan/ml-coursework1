import time

import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Import dataset
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dataset = pandas.read_csv("diabeticret_dataset.csv")

# Preprocessing the dataset: Normalizing Values
normalizer = MinMaxScaler()
dataset[['C', 'D', 'E',	'F', 'G', 'H', 'I',	'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']] = normalizer.fit_transform(dataset[['C', 'D', 'E',	'F', 'G', 'H', 'I',	'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']])
X = dataset.drop(columns=['T'])
Y = dataset['T']

# Split test data from dataset
X, X_test, Y, Y_test = train_test_split(X, Y, test_size= 20/100)

# @exec_model: Execute and print out the results of the given model
def exec_model(model, modelname):
    start = time.time()
    model.fit(X, Y);
    accuracy = round(accuracy_score(Y_test, model.predict(X_test))* 100, 2)
    stop = time.time()
    training_time = round((stop - start), 4)
    return accuracy, training_time

# returns the average of a list
def Average(lst):
    return sum(lst) / len(lst)

# List of algorithms to test
MODELS = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVC" : SVC(kernel='linear')
}

# Run the algorithms for 500 times and print the averages
for modelname, model in MODELS.items():
    accuracy_list = []
    training_time_list = []
    for index in range(500):
        accuracy , training_time = exec_model(model, modelname)
        accuracy_list.append(accuracy)
        training_time_list.append(training_time)

    accuracy_average = Average(accuracy_list)
    time_average = Average(training_time_list)

    print(f"#####-{modelname}-#####")
    print("Average Accuracy: ", round(accuracy_average, 2))
    print(f"Average Time:", round(time_average, 4), "s")
    print("---------")
