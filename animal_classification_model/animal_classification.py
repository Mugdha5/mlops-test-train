#The Zoo dataset (https://www.kaggle.com/uciml/zoo-animal-classification)
#This dataset consists of 101 animals from a zoo.
#There are 16 features or attributes with various traits to describe the animals:
# animal_name, hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone,
# breathes, venomous, fins, legs, tail, domestic, catsize.
#The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate
#The purpose for this dataset is to be able to predict the classification of the animals, based upon the variables

# Import pandas library
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib

dataset = pd.read_csv('animal_classification_model/zoo.csv')

# The first column which are names of the animals is not a Feature that can be used.
# Extract the column and save for later.
animal_names = dataset['animal_name'].tolist()

# Drop the first column which are the animal names
dataset=dataset.drop('animal_name',axis=1)
# The given dataset is split into two  parts - training and testing set, in a ratio.

# copy all columns excluding last column
X = dataset.loc[:, dataset.columns != 'class_type']

# copy the last column only
Y = dataset['class_type']

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, stratify=Y, test_size= 0.3)

# The model we are choosing is a Random Forest Classifier model.
model = RandomForestClassifier()

# Train the model
model.fit(X_train, Y_train)

# Predict the values for testing set
Y_predict = model.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
print(acc * 100 , "Accuracy Score of RandomForestClassifier before Hyper Parameter Tuning- %")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    #'max_terminal_nodes': [2,1]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, Y_train)
# 5. Evaluate
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score (Accuracy):", grid_search.best_score_*100)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Set Accuracy:", accuracy_score(Y_test, y_pred)*100)

joblib.dump(model,'animal_classification_model.joblib')
