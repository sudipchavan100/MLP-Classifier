import os
import sys
import ast

ECG_dir = "RECOLA/RECOLA-Biosignals-features/ECG"
EDA_dir = "RECOLA/RECOLA-Biosignals-features/EDA"
#dir_path = os.path.dirname(os.path.realpath(__file__))
arousal_dir = "RECOLA/RECOLA-Annotation/emotional_behaviour/arousal"
valence_dir = "RECOLA/RECOLA-Annotation/emotional_behaviour/valence"

ECG_files = []
EDA_files = []
arousal_files = []
valence_files = []

for root, dirs, files in os.walk(ECG_dir):
    for file in files:
        if file.endswith(".arff"):
            #print(os.path.join(root, file))
            ECG_files.append(file)
for root, dirs, files in os.walk(EDA_dir):
    for file in files:
        if file.endswith(".arff"):
            #print(os.path.join(root, file))
            EDA_files.append(file)
for root, dirs, files in os.walk(arousal_dir):
    for file in files:
        if file.endswith(".csv"):
            #print(os.path.join(root, file))
            arousal_files.append(file)
for root, dirs, files in os.walk(valence_dir):
    for file in files:
        if file.endswith(".csv"):
            #print(os.path.join(root, file))
            valence_files.append(file)            
#print(ECG_files)
#print(EDA_files)
ECG_files = ["P16.arff"]
EDA_files = ["P16.arff"]
arousal_files = ["P16.csv"]
valence_files = ["P16.csv"]
ECG_data = {}
for file in ECG_files:
    with open(ECG_dir + "/" + file) as f:
        lines = f.readlines()
    
    for line in lines:
        if line == "" or line.startswith("@"):
            continue
        
        values = line.strip("\n").split(",") 
        ECG_data[file.split(".")[0]+"_"+values[0]]=[ast.literal_eval(x) for x in values[1:]]

EDA_data = {}
for file in EDA_files:
    with open(EDA_dir + "/" + file) as f:
        lines = f.readlines()
    
    for line in lines:
        if line == "" or line.startswith("@"):
            continue
        
        values = line.strip("\n").split(",") 
        EDA_data[file.split(".")[0]+"_"+values[0]]=[ast.literal_eval(x) if x != "?" else 0 for x in values[1:]]

arousal_data = {}
for file in arousal_files:
    with open(arousal_dir + "/" + file) as f:
        lines = f.readlines()
    
    for line in lines:
        if line == "" or line == None:
            continue
        
        values =line.strip("\n").split(";",1)
        arousal_data[file.split(".")[0]+"_"+values[0]]=values[1]
        
valence_data = {}
for file in valence_files:
    with open(valence_dir + "/" + file) as f:
        lines = f.readlines()
    
    for line in lines:
        if line == "" or line == None:
            continue
        
        values =line.strip("\n").split(";",1)
        valence_data[file.split(".")[0]+"_"+values[0]]=values[1]
        
X_data_arousal = []
y_data_arousal = []
X_data_valence = []
y_data_valence = []

for key in arousal_data.keys():
    if(key not in ECG_data.keys() or key not in EDA_data.keys()):
        continue
    X_data_arousal.append(ECG_data[key] + EDA_data[key])
    y_data_arousal.append(arousal_data[key])
              
for key in valence_data.keys():
    if(key not in ECG_data.keys() or key not in EDA_data.keys()):
        continue
    X_data_valence.append(ECG_data[key] + EDA_data[key])
    y_data_valence.append(valence_data[key])
                    
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(X_data_arousal, y_data_arousal, random_state=2)

y_train_1 = list(set(y_train))
y_train_index = [y_train_1.index(x) for x in y_train]

y_test_1 = list(set(y_test))
y_test_index = [y_test_1.index(x) for x in y_test]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train_index)
print('Number of layers: %s. Number of outputs: %s' % (clf.n_layers_, clf.n_outputs_))
predictions = clf.predict(X_test)
print('Accuracy:', clf.score(X_test, y_test_index) )
for i, p in enumerate(predictions[:10]):
    print('True: %s, Predicted: %s' % (y_test[i], p))
    
    
    