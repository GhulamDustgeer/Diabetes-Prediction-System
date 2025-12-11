# -------------------------------------------------------
# 1. Import Required Libraries
# -------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# 2. Load the Diabetes Dataset
# -------------------------------------------------------
data_set = pd.read_csv('diabetes.csv')

# Display first 5 rows of the dataset
print(data_set.head())

# -------------------------------------------------------
# 3. Show Statistical Summary of the Data
# -------------------------------------------------------
print(data_set.describe())

# -------------------------------------------------------
# 4. Check Class Distribution (0 = Non-diabetic, 1 = Diabetic)
# -------------------------------------------------------
print(data_set['Outcome'].value_counts())

# -------------------------------------------------------
# 5. Show Average Values for Each Class
# -------------------------------------------------------
print(data_set.groupby('Outcome').mean())

# -------------------------------------------------------
# 6. Split Features (X) and Target (Y)
# -------------------------------------------------------
X = data_set.drop(columns='Outcome', axis=1)
Y = data_set['Outcome']

print(X)
print(Y)

# -------------------------------------------------------
# 7. Visualize Class Distribution
# -------------------------------------------------------
sns.countplot(x=Y)
plt.title("Class Distribution of Outcome")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

# -------------------------------------------------------
# 8. Standardize the Feature Values
# -------------------------------------------------------
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)

X = standardized_data
Y = data_set['Outcome']

print(standardized_data)

# -------------------------------------------------------
# 9. Split Data into Train and Test Sets
# -------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# 10. Train SVM Classifier with Linear Kernel
# -------------------------------------------------------
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# -------------------------------------------------------
# 11. Evaluate Model Accuracy for Training Data
# -------------------------------------------------------
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# -------------------------------------------------------
# 12. Evaluate Model Accuracy for Testing Data
# -------------------------------------------------------
x_test_prediction = classifier.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy score of the testing data : ', testing_data_accuracy)

# -------------------------------------------------------
# 13. Predict Diabetes for a Single Input
# -------------------------------------------------------
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)

# Convert input to numpy array
input_array = np.asarray(input_data)

# Reshape since prediction expects 2D array
input_reshaped = input_array.reshape(1, -1)

# Standardize input
std_data = scalar.transform(input_reshaped)

print(std_data)

# Make Prediction
prediction = classifier.predict(std_data)
print(prediction)

# -------------------------------------------------------
# 14. Display Final Result
# -------------------------------------------------------
if prediction[0] == 0:
    print("The person is **not diabetic**")
else:
    print("The person **is diabetic**")
