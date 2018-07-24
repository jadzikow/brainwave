import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import brainwave

classifier, X_min, X_max = brainwave.create_classifier()
test_X, test_Y = brainwave.load_data("data/6way/test", ["idle", "look_right", "look_left", "eyes_closed", "jaw_clench", "smile"], is_directory=True, interval_seconds=2, step_size=32)
predictions = brainwave.preprocess_and_classify(classifier, test_X, X_min, X_max)
print(predictions)
print(test_Y)
print(np.mean(np.equal(predictions, test_Y)))
