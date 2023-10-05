#-------------------------------------------------------------------------
# AUTHOR: Cleighton Greer
# FILENAME: knn.py
# SPECIFICATION: computes error rate of 1-nn model using cross validation method
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

#loop your data to allow each instance to be your test set
incorrect_predictions = 0
total_predictions = len(db)
for i in range(0, len(db)):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    ##Get each instance and extract it's feature values (AS FLOATS)
        #skip instance of index i 
    print(f"\nindex i={i} and test instance={db[i]}")
    X = ([[float(enum_feature_vector[1][0]), float(enum_feature_vector[1][1])] for enum_feature_vector in enumerate(db) if enum_feature_vector[0] != i])
        #(Above only work for two feature values)
    # print(f"\n{X}")
       

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    ##Do same as above but parse the last column (class label column)
    Y = ([enum_feature_vector[1][2] for enum_feature_vector in enumerate(db) if enum_feature_vector[0] != i])
    ##Convert class labels for float values
    set_of_class_labels = set(Y)
    class_label_integer_val_mappings = dict([[int_val_pair[1], int_val_pair[0]] for int_val_pair in enumerate(set_of_class_labels)])
    Y = [class_label_integer_val_mappings[class_label] for class_label in Y]
       

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(db[i][0]), float(db[i][1])]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    true_label = class_label_integer_val_mappings[db[i][2]]
    print(f"true class label: {true_label}, predicted class label: {class_predicted}\nCorrect prediction: {true_label == class_predicted}")

    if (true_label != class_predicted):
       incorrect_predictions += 1

#print the error rate
print(f"\n(1-NN Model)Error rate: {incorrect_predictions/total_predictions}")






