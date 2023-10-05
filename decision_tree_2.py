#-------------------------------------------------------------------------
# AUTHOR: Cleighton Greer
# FILENAME: decision_tree_2.py
# SPECIFICATION: create pre-pruned decision tree from input training data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    list_of_column_key_val_mappings = [{'Prepresbyopic': 0, 'Presbyopic': 1, 'Young': 2},
                                        {'Hypermetrope': 0, 'Myope': 1},
                                        {'No': 0, 'Yes': 1},
                                        {'Reduced': 0, 'Normal': 1},
                                        {'Yes': 0, 'No': 1}]

    ##Transform data (feature values) using generated column key/value mapping
    #iterate through every row minus
    for row in range(0, len(dbTraining)):
        new_row_of_X = []
        for col in range(0, len(dbTraining[0])-1): #skip last column which is the class label
            new_row_of_X.append(list_of_column_key_val_mappings[col][dbTraining[row][col]]) # mappings: [which column][key] = corresponding integer value
        X.append(new_row_of_X) #numericalized instance

    # print(X)

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    ##Do same as above but exclusively for the last column of the data
    ##Transform class label values using generated column key/value mapping
    for row in range(0, len(dbTraining)):
        Y.append(list_of_column_key_val_mappings[-1][dbTraining[row][-1]])

    # print(Y)

    #loop your training and test tasks 10 times here
    print("\n////////////////////////////////////")
    print(f"TRAINING DATASET: {ds}")
    print(f"TEST DATASET: contact_lens_test.csv")
    list_of_accuracy_results = []
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        dbTest = []
        test_data_csv = open("contact_lens_test.csv", "r")
        csv_reader = csv.reader(test_data_csv)
        for j, row in enumerate(csv_reader):
            if j > 0: #skipping the header
                dbTest.append (row)
        test_data_csv.close()

        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        test_X = [] #measured attributes of instances
        test_Y = [] #contains true label of instances

        ##Transform data (feature values) using generated column key/value mapping
        #iterate through every row minus
        for row in range(0, len(dbTest)):
            new_row_of_X = []
            for col in range(0, len(dbTest[0])-1): #skip last column which is the class label
                new_row_of_X.append(list_of_column_key_val_mappings[col][dbTest[row][col]]) # mappings: [which column][key] = corresponding integer value
            test_X.append(new_row_of_X) #numericalized instance

        ##Do same as above but exclusively for the last column of the data
        ##Transform class label values using generated column key/value mapping
        for row in range(0, len(dbTest)):
            test_Y.append(list_of_column_key_val_mappings[-1][dbTest[row][-1]])

        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        correct_predictions = 0
        total_prediction = len(test_Y)
        for k in range(0, len(test_Y)): #do prediction and evaluation
            prediction = clf.predict([test_X[k]])[0]
            if (prediction == test_Y[k]):
                correct_predictions += 1
        accuracy = correct_predictions/total_prediction
        list_of_accuracy_results.append(accuracy)
        print(f"Iteration:{i} |||| Accuracy: {accuracy}")

    #find the average of this model during the 10 runs (training and test set)
    avg_accuracy = 0
    for accuracy in list_of_accuracy_results:
        avg_accuracy+= accuracy
    avg_accuracy = avg_accuracy/len(list_of_accuracy_results)

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"\nModel Trained From: {ds} |||| Final (Avg) Accuracy: {avg_accuracy}\n")

 
