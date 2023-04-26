import os,time
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import numpy as np
from matplotlib.figure import Figure


    # Importing image data file:
data = loadmat("D:/IIT Bombay  CSRE/sem2/GNR 602 - ASIP/project/Group6-Abhishek-Dheeraj-Surbhi/nn_bagging/Indian_pines_corrected.mat")
data_name=next(reversed(data))
data=data[data_name]
    # Importing ground truth file:
label = loadmat("D:/IIT Bombay  CSRE/sem2/GNR 602 - ASIP/project/Group6-Abhishek-Dheeraj-Surbhi/nn_bagging/Indian_pines_gt.mat")
label_name=next(reversed(label))
label=label[label_name]
# Flatten the data to 2D array
X = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
y = label.ravel()
# Split the dataset into training and testing sets
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=42)
# Perform Min-Max scaling on input features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_total_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
   


global fig
global y_pred_ovr,y_test_ovr,training_time_ovr,Time_taken_ovr
global y_pred_ovo,y_test_ovo,training_time_ovo,Time_taken_ovo
# Getting user defined kernel_type and slack parameter:
kernel='linear'
C=1

# Define the SVM model with non-linear kernel and varying slack parameter C
svm_model = SVC(kernel=kernel, C=C)

# Fit the SVM model with the one versus one method
start_time = time.time()
svm_ovo = OneVsOneClassifier(svm_model).fit(X_train_scaled, y_train)
Time_taken_ovo = time.time() - start_time
# Predict the labels for test data:
y_test_ovo = svm_ovo.predict(X_test_scaled) 
# testing on whole dataset for visualization
y_pred_ovo = svm_ovo.predict(X_total_scaled) 

# Fit the SVM model with the one versus rest method
start_time1 = time.time()
svm_ovr = OneVsRestClassifier(svm_model).fit(X_train_scaled, y_train)
Time_taken_ovr = time.time() - start_time1
# Predict the labels for test data:
y_test_ovr = svm_ovr.predict(X_test_scaled) 

# testing on whole dataset for visualization
y_pred_ovr = svm_ovr.predict(X_total_scaled)

# Reshaping :
y_champ_ovo=y_pred_ovo.reshape(data.shape[0],data.shape[1])
y_champ_ovr=y_pred_ovr.reshape(data.shape[0],data.shape[1])

from sklearn.metrics import accuracy_score
acc_ovo = accuracy_score(y_test, y_test_ovo)
acc_ovr = accuracy_score(y_test, y_test_ovr)

# Print the results
print("Accuracy (One-vs-One): {:.2f}%".format(acc_ovo * 100))
print("Accuracy (One-vs-Rest): {:.2f}%".format(acc_ovr * 100))
# Print the time taken for one-vs-one strategy
print("Training time (One-vs-One): {:.2f} seconds".format(Time_taken_ovo))
# Print the time taken one-vs-rest strategy
print("Training time (One-vs-Rest): {:.2f} seconds".format(Time_taken_ovr))

# Calculate confusion matrix for one-vs-one strategy
confusion_matrix_ovo = np.zeros((label.max(), label.max()), dtype=int)
for i in range(len(y_test)):
    confusion_matrix_ovo[y_test[i]-1][y_test_ovo[i]-1] += 1
# Calculate confusion matrix for one-vs-rest strategy
confusion_matrix_ovr = np.zeros((label.max(), label.max()), dtype=int)
for i in range(len(y_test)):
    confusion_matrix_ovr[y_test[i]-1][y_test_ovr[i]-1] += 1

# Calculate overall accuracy for one-vs-one and one-vs-rest strategies
total = np.sum(confusion_matrix_ovo)
correct_ovo = np.sum(np.diag(confusion_matrix_ovo))
overall_acc_ovo = correct_ovo / total
correct_ovr = np.sum(np.diag(confusion_matrix_ovr))
overall_acc_ovr = correct_ovr / total

# Calculate user's accuracy, producer's accuracy, omission error, and commission error for one-vs-one strategy
user_acc_ovo = np.zeros((label.max(),))
prod_acc_ovo = np.zeros((label.max(),))
omission_error_ovo = np.zeros((label.max(),))
commission_error_ovo = np.zeros((label.max(),))
for i in range(label.max()):
    user_acc_ovo[i] = confusion_matrix_ovo[i,i] / np.sum(confusion_matrix_ovo[i,:])
    prod_acc_ovo[i] = confusion_matrix_ovo[i,i] / np.sum(confusion_matrix_ovo[:,i])
    omission_error_ovo[i] = 1 - user_acc_ovo[i]
    commission_error_ovo[i] = 1 - prod_acc_ovo[i]

# Calculate user's accuracy, producer's accuracy, omission error, and commission error for one-vs-rest strategy
user_acc_ovr = np.zeros((label.max(),))
prod_acc_ovr = np.zeros((label.max(),))
omission_error_ovr = np.zeros((label.max(),))
commission_error_ovr = np.zeros((label.max(),))
for i in range(label.max()):
    user_acc_ovr[i] = confusion_matrix_ovr[i,i] / np.sum(confusion_matrix_ovr[i,:])
    prod_acc_ovr[i] = confusion_matrix_ovr[i,i] / np.sum(confusion_matrix_ovr[:,i])
    omission_error_ovr[i] = 1 - user_acc_ovr[i]
    commission_error_ovr[i] = 1 - prod_acc_ovr[i]

print(f'overall Accuracy (one vs one) : {overall_acc_ovo}')
print(f'overall Accuracy (one vs rest) : {overall_acc_ovr}')
print(f'user\'s accuracy (one vs one) : {user_acc_ovo}')
print(f'Producer\'s accuracy (one vs one) : {prod_acc_ovo}')
print(f'user\'s accuracy (one vs rest) : {user_acc_ovr}')
print(f'Producer\'s accuracy (one vs rest) : {user_acc_ovr}')
print(f'Omission error (one vs one) : {omission_error_ovo}')
print(f'Commission error (one vs one) : {commission_error_ovo}')
print(f'Omission error (one vs rest) : {omission_error_ovr}')
print(f'Commission error (one vs rest) : {commission_error_ovr}')







