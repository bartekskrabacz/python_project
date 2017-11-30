from __future__ import print_function

import logging
from Data_importer import Data_importer
from Drawer import Drawer
from Categorizer import Categorizer
from AlgorithmImplementer import AlgorithmImplementer

class Runner:

    def __init__(self):
        pass

    def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

        data_importer = Data_importer()
        lfw_people = data_importer.import_data(min_faces_per_person=70, resize=0.4)

        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]

        categorizer = Categorizer(lfw_people)
        n_samples, h, w = categorizer.return_shape_data()

        X_train, X_test, y_train, y_test = categorizer.categorize()

        executor = AlgorithmImplementer( X_train,X_test,y_train,y_test)
        y_pred,eigenfaces = executor.executeAlgorithm(h,w,target_names,n_classes)

        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        drawer = Drawer(y_pred, y_test, target_names, X_test, eigenfaces)
        drawer.show(h,w)
        data_importer.clear_cache()
