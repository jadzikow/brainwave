import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import brainwave

#classifier, X_min, X_max = brainwave.create_classifier()
#class_names = ["idle", "look_right", "look_left", "eyes_closed", "jaw_clench", "smile"]
class_names = ["idle", "look_right", "look_left", "jaw_clench", "smile"]
classifier = brainwave.Classifier(class_names, train_from_scratch=True)

#class_names = ["idle", "up", "down"]
#test_X, test_Y = brainwave.load_data("data/6way/test", class_names, is_directory=True, classes_in_subdirs=True, interval_seconds=2, step_size=32)
test_X, test_Y = brainwave.load_data("data/5way/test", class_names, is_directory=True, classes_in_subdirs=True, interval_seconds=1, step_size=32)
#test_X, test_Y = brainwave.load_data("data/mental/test", class_names, is_directory=True, classes_in_subdirs=True, interval_seconds=2, step_size=32)
#predictions = brainwave.preprocess_and_classify(classifier, test_X, X_min, X_max)
predictions = classifier.preprocess_and_classify(test_X)

print(predictions)
print(test_Y)
print(np.mean(np.equal(predictions, test_Y)))

confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
for i in range(len(test_Y)):
    confusion_matrix[int(test_Y[i]), int(predictions[i])] += 1

print(confusion_matrix)
    
