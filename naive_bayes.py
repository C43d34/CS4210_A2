#-------------------------------------------------------------------------
# AUTHOR: Cleighton Greer
# FILENAME: naive_bayes.py
# SPECIFICATION: classify data using naive_bayes method 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2.5hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv 

#reading the training data in a csv file
training_data = []
csv_file_obj = open("weather_training.csv", "r")
csv_reader = csv.reader(csv_file_obj)
for i, line in enumerate(csv_reader):
    if i > 0: #skip header row
        training_data.append(line)
csv_file_obj.close()


#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]

##Capture unique set of features and map to integer values
    #Ensure we don't include unique identifier "feature" in column 1 of training data
    #Don't include class label at column n as part of features
list_of_column_attribute_mappings_to_integer = [{'Sunny': 0, 'Overcast': 1, 'Rain': 2},
                                                {'Hot': 0, 'Mild': 1, 'Cool': 2},
                                                {'Normal': 0, 'High': 1},
                                                {'Strong': 0, 'Weak': 1}]  
#Apply mapping of features to integer values and store in X
X = []
for row in range(0, len(training_data)):
    row_feature_value_vec = []
    for col in range(0, len(list_of_column_attribute_mappings_to_integer)):
        row_feature_value_vec.append(list_of_column_attribute_mappings_to_integer[col][training_data[row][col+1]])
    X.append(row_feature_value_vec)
# print(X)



#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#create mappings for these labels
label_mappings = {'Yes': 0, 'No': 1} #goes from class label to integer
inverted_label_mappings = {0: 'Yes', 1: 'No'} #goes from integer to class label 
# print(label_mappings)
Y = [label_mappings[entry[-1]] for entry in training_data]
# print(Y)

# X = [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 1], [2, 2, 0, 1], [2, 1, 1, 1], [2, 1, 1, 0], [1, 1, 1, 0], [0, 2, 0, 1], [0, 1, 1, 1], [2, 2, 1, 1], [0, 2, 1, 0], [1, 2, 0, 0], [1, 0, 1, 1], [2, 2, 0, 0]] 
# Y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
test_data = []
header = []
csv_file_obj = open("weather_test.csv", "r")
csv_reader = csv.reader(csv_file_obj)
for i, line in enumerate(csv_reader):
    if i > 0: #skip header row
        test_data.append(line)
    else:
        line.append("Confidence")
        header = line
csv_file_obj.close()

#printing the header os the solution
print(f"{' | '.join(header)}")
#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
##Convert feature columns of test data to integer just like training data
    #skip first column and last column (ID and to-be-assigned class label)
# test_data_x = [[0, 0, 1, 1],[0, 0, 1, 0], [0, 1, 0, 1],[1, 0, 0, 0],[1, 1, 0, 1],[1, 1, 0, 0],[2, 2, 1, 0],[2, 0, 1, 0],[2, 0, 0, 1],[2, 2, 0, 0]]
for row in range(0, len(test_data)):
    test_instance = [[column_attribute_mapping[test_data[row][col+1]] for col, column_attribute_mapping in enumerate(list_of_column_attribute_mappings_to_integer)]]
    prediction_vector = clf.predict_proba(test_instance)[0]

    #get highest probable value from vector
    most_probable_value = -1
    class_of_most_probable_value = ""
    for i in range(0, len(prediction_vector)):
        if (most_probable_value < prediction_vector[i]):
            most_probable_value = prediction_vector[i]
            class_of_most_probable_value = inverted_label_mappings[i]

    if (most_probable_value >= 0.75): #only showcase predictions of high confidence 
        print(f"{' | '.join(test_data[row][0:-1])} | {class_of_most_probable_value} | {most_probable_value:.2f}")

