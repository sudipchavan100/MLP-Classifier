The "Metadata" and the "Recordings" folders are not important. The recordings are the raw ECG and EDA (skin conductance) signals.

We are only interested in the features that were extracted from the signals. The features are store in the folder "Features". Each file is a patient (e.g. P16), and each patient contains a list of features (e.g., timestamp, HR heart rate, HRV heart rate variance, FFT Furier Transform, etc.). Each row in that file is the set of features at a certain time. This is going to be the input of your machine learning algorithm.

The output of the machine learning algorithm should be the values of the valence and arousal at the same time stamp as the input. You should read the expected values at the same time as the input features, and these "target" values are stored in the "RECOLA-Annotation\emotional_behaviour" folder.

Your classifier, in a first implementation, can be a Neural Network (Multi-layer perceptron, NOT the Multi-layer perceptron classifier) that has 2 outputs that are real numbers: one is the estimated value of the valence, one is the arousal.

You can use the sklearn library for Python. You can have a look here: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

You will find it easier to start with just one patient (e.g., P16). You can take all the approx. 7400 samples, and you can divide them into a training and test set. You can put 80% of the 7400 samples into the training set, and the remaining 20% in the test.

You can start with two layers, and you can put 100-200 neurons per layer.
