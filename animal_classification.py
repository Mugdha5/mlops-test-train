#The Zoo dataset (https://www.kaggle.com/uciml/zoo-animal-classification)
#This dataset consists of 101 animals from a zoo.
#There are 16 features or attributes with various traits to describe the animals:
# animal_name, hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone,
# breathes, venomous, fins, legs, tail, domestic, catsize.
#The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate
#The purpose for this dataset is to be able to predict the classification of the animals, based upon the variables

# Import pandas library
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, explained_variance_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Start MLflow experiment
mlflow.set_experiment("Random Model Classifier")

with mlflow.start_run():

    dataset = pd.read_csv('./zoo.csv')
    
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
    
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, stratify=Y, test_size= 0.2)
    
    # The model we are choosing is a Random Forest Classifier model.
    model = RandomForestClassifier()
    
    # Train the model
    model.fit(X_train, Y_train)
    
    # Predict the values for testing set
    Y_predict = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_predict)
    print(acc * 100 , "Accuracy Score of RandomForestClassifier before Hyper Parameter Tuning- %")
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 20, 50],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 4],
        #'max_terminal_nodes': [2,1]
        'max_leaf_nodes': [10, 20]
        
    }
    strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
   # grid_search = GridSearchCV(
    #    estimator=model,
    #    param_grid=param_grid,
    #    scoring='accuracy',
    #    cv=25,
    #    n_jobs=-1,
    #    verbose=1
    #)
    grid_search.fit(X_train, Y_train)
    
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # 5. Evaluate
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score (Accuracy):", grid_search.best_score_*100)
    print("Test Set Accuracy:", accuracy_score(Y_test, y_pred)*100)
    print(f"Test Accuracy: {best_model.score(X_test, Y_test) * 100:.2f}%")
    print(f"Train Accuracy: {best_model.score(X_train, Y_train) * 100:.2f}%")
    # Create the confusion matrix
    conf_matrix = confusion_matrix(Y_test, y_pred)

    print(classification_report(Y_test, y_pred))
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the figure to a file
    plt.savefig("confusion_matrix.png")
    
    
    
    
    # Log classification metrics 
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(Y_test, y_pred))
    #mlflow.log_metric("confusion_matrix", confusion_matrix(Y_test, y_pred).ravel().tolist())  # Log as list
    # Log parameters and metrics
    #mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_param("n_samples", len(X_train))
    mlflow.sklearn.log_model(best_model, "Random Model Classifier")
    # Log the image as an artifact
    mlflow.log_artifact("confusion_matrix.png")


    # Save model
    mlflow.sklearn.log_model(model, "Random Model Classifier")

print("Experiment Logged Successfully!")

joblib.dump(model,'animal_classification_model.joblib')

