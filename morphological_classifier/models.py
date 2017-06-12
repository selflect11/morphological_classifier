# TRAINING: Loads training.txt into a Text() data structure
#           Gets the list of WordArrays and list of Target classes
#           Feeds X,Y to fit
# TESTING:  Tries to predict the test.txt
# CROSS VALIDATION: TODO
import numpy as np
from sklearn.neural_network import MLPClassifier

# training
training_filepath = ''
train_text = Text()
train_text.read_file(training_filepath)
X, y = train_text.get_data()
# Trains. Gets the classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_size=(5,2), random_state=1)
clf.fit(X,Y)

# tests
testing_filepath = ''
test_text = Text()
test_text.read_file(testing_filepath)
clf.predict(word)
