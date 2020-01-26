import itertools
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas
import librosa
import jinja2
import sklearn
from src import configuration
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def confusion_matrix(cm, classes,
                     title='Confusion matrix',
                     cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_cnf(model, dataset_x, dataset_y, GENRES):
    true_y = dataset_y
    true_x = dataset_x
    pred = model.predict(true_x)

    print("---------------Анализ модели----------------\n")

    # print("Настоящие лейблы тестовой выборки: \n{}\n".format(true_y))
    # print("Предсказанные лейблы тестовой выборки: \n{}".format(pred))

    cnf_matrix = sklearn.metrics.confusion_matrix(true_y, pred)
    plt.figure()
    a = confusion_matrix(cnf_matrix, classes=GENRES, title='Confusion matrix')

def main():
    data_set=pandas.read_csv('dataset.csv',index_col=False)
    GENRES=['metal', 'classical', 'hiphop', 'blues', 'pop',
            'reggae', 'country', 'disco', 'jazz', 'rock']

    number_of_rows,number_of_cols = data_set.shape
    data_set[:5].style

    data_set_values=numpy.array(data_set)

    train, test = train_test_split(data_set_values, test_size = 0.85,random_state=2,
                                  stratify=data_set_values[:,number_of_cols-1])

    train_x=train[:,:number_of_cols-1]
    train_y=train[:,number_of_cols-1]

    test_x=test[:,:number_of_cols-1]
    test_y=test[:,number_of_cols-1]

    print("Размер обучающей выборки: {}".format(train.shape))
    print("Размер тестовой выборки: {}".format(test.shape))


    results_knn = []
    for i in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x, train_y)
        results_knn.append(knn.score(test_x, test_y))

    max_accuracy_knn = max(results_knn)
    best_k = 1 + results_knn.index(max(results_knn))
    print("Точность: {:.3f} на тестовой выборке с {} -neighbors.\n".format(max_accuracy_knn, best_k))

    plt.plot(numpy.arange(1, 11), results_knn)
    plt.xlabel("n Neighbors")
    plt.ylabel("Accuracy")
    plt.show()

    print("\n\nМодель K-Neighbors Classifier")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_x, train_y)
    print("Обучающая: {:.3f}".format(knn.score(train_x, train_y)))
    print("Тестовая: {:.3f}".format(knn.score(test_x, test_y)))

    plot_cnf(knn, test_x, test_y, GENRES)


    results_forest = []
    for i in range(2, 20):
        forest = RandomForestClassifier(random_state=42, n_estimators=i)
        forest.fit(train_x, train_y)
        results_forest.append(forest.score(test_x, test_y))

    max_accuracy_forest = max(results_forest)
    best_n_est = 2 + results_forest.index(max(results_forest))
    print("Точность: {:.3f} на тестовой выборке с {} -estimators.\n".format(max_accuracy_forest, best_n_est))

    plt.plot(numpy.arange(2, 20), results_forest)
    plt.xlabel("n Estimators")
    plt.ylabel("Accuracy")
    plt.show()

    print("\n\nМодель Random Forest Classifier")
    forest = RandomForestClassifier(random_state=42, n_estimators=best_n_est)
    forest.fit(train_x, train_y)
    print("Обучащая: {:.3f}".format(forest.score(train_x, train_y)))
    print("Тестовая: {:.3f}".format(forest.score(test_x, test_y)))

    plot_cnf(forest, test_x, test_y, GENRES)

    print("\n\nМодель SVM")
    svm=SVC(C=100,gamma=0.08)
    svm.fit(train_x,train_y)
    print("Обучающая: {:.3f}".format(svm.score(train_x,train_y)))
    print("Тестовая: {:.3f}".format(svm.score(test_x,test_y)))

    plot_cnf(svm,test_x,test_y,GENRES)

    print("\n\nМодель MLP Classifier")
    neural=MLPClassifier(max_iter=400,random_state=2,hidden_layer_sizes=[40,40])
    neural.fit(train_x,train_y)
    print("Обучающая: {:.3f}".format(neural.score(train_x,train_y)))
    print("Тестовая: {:.3f}".format(neural.score(test_x,test_y)))

    plot_cnf(neural,test_x,test_y,GENRES)

if __name__ == '__main__':
    main()
