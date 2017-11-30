from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class AlgorithmImplementer:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test



    def executeAlgorithm(self,h,w,target_names,n_classes):
        n_components = 150

        # print("Extracting the top %d eigenfaces from %d faces"
        #       % (n_components, self.X_train.shape[0]))
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(self.X_train)

        # print("done in %0.3fs" % (time() - t0))
        eigenfaces = pca.components_.reshape((n_components, h, w))

        # print("Projecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        X_train_pca = pca.transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        # print("done in %0.3fs" % (time() - t0))

        # #############################################################################
        # Train a SVM classification model

        # print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_pca, self.y_train)
        # print("done in %0.3fs" % (time() - t0))
        # print("Best estimator found by grid search:")
        # print(clf.best_estimator_)

        # #############################################################################
        # Quantitative evaluation of the model quality on the test set

        # print("Predicting people's names on the test set")
        t0 = time()
        y_pred = clf.predict(X_test_pca)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(self.y_test, y_pred, target_names=target_names))
        print(confusion_matrix(self.y_test, y_pred, labels=range(n_classes)))

        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        return y_pred,eigenfaces