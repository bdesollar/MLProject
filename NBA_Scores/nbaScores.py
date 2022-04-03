# imports
import numpy as np
from mlwpy import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from sklearn import (datasets, neighbors,
                     naive_bayes,
                     model_selection as skms,
                     linear_model, dummy,
                     metrics,
                     pipeline,
                     preprocessing as skpre) 
import csv
from sklearn import tree


data_train_df = pd.read_csv("train.csv") 
data_train_ft = data_train_df.drop('PTS', axis=1)
data_train_tgt = data_train_df["PTS"]



# initial exploration of data
print("data_train_df:")
display(data_train_df.head(10))

# look for columns with missing data
print("Info gives us this:")
data_train_df.info()

# let's focus on only features that seem most useful for now
features = ['HEIGHT',
            'SEASON_EXP',
            'DRAFT_ROUND',
            'AST',
            'REB',
            'ALL_STAR_APPEARANCES',
            # 'Avg_Utilization_Ratio',
            # 'Total_Trans_Ct',
            # 'Total_Ct_Chng_Q4_Q1', 
           # 'Total_Revolving_Bal',
           # 'Total_Amt_Chng_Q4_Q1',
            ]

# add survival column

# titanic_train_df = titanic_train_df[features + ['Survived']]

# for now, let's use a simple approach to estimate the age (and consider revisiting this estimate later)
# when it is missing by using the median from the other samples

# median_age = data_train_df['Customer_Age'].median() # note: by default, this will skip NA/null values
# print(f'Median age: {median_age:.2f}')
# data_train_df['Customer_Age'] = data_train_df['Customer_Age'].fillna(median_age)

# convert sex to a 0/1 category
# data_train_df['Gender'].replace(['M','F'],[0,1],inplace=True)

# display updated table
print("data_train_df:")
display(data_train_df.head(10))
print("data_train_df.info():")
display(data_train_df.info())
# print("describe:")
# display(data_train_df.describe())

# also convert to floating type for scaler
# data_train_df = data_train_df[features].astype(float) 
# titanic_train_tgt = titanic_train_df['Survived']

# use cross-validation to pick best model (use accuracy since 
# kaggle will evaluate on accuracy as well)
models_to_try = {'nb': naive_bayes.GaussianNB()}
# add k-NN models with various values of k to models_to_try
for k in range(1,42,2):
    models_to_try[f'{k}-NN'] = neighbors.KNeighborsClassifier(n_neighbors=k)

# scaler = skpre.StandardScaler()
pipelines_to_try = \
    {'GNB0' : naive_bayes.GaussianNB(),

     # 'SVC(1)' : svm.SVC(kernel="linear"),
     #'SVC(2)' : svm.LinearSVC(),
     #'SVC(3)' : svm.SVC(kernel="poly" ,C=.8),
     #'SVC(4)' : svm.NuSVC(kernel='linear', nu=.2),
     'DTC' : tree.DecisionTreeClassifier(),
     'DTC-5' : tree.DecisionTreeClassifier(max_depth=5),
     'DTC-10' : tree.DecisionTreeClassifier(max_depth=10),
     '5NN-C' : neighbors.KNeighborsClassifier(),
     '10NN-C' : neighbors.KNeighborsClassifier(n_neighbors=10)}

baseline = dummy.DummyClassifier(strategy="uniform")
'''
for model_name in models_to_try:
    pipelines_to_try[f'std_{model_name}_pipe'] = pipeline.make_pipeline(scaler, 
                                                      models_to_try[model_name])
'''

sv_classifiers = {"SVC(Linear)"   : svm.SVC(kernel='linear'),
                  "NuSVC(Linear)" : svm.NuSVC(kernel='linear', nu=.9)} 


accuracy_scores = {}
for name, model in pipelines_to_try.items():
    #loo = skms.LeaveOneOut()s
    scores = skms.cross_val_score(model,
                                  data_train_ft[features],
                                  data_train_tgt,
                                  #cv=loo,
                                  cv=10,
                                  scoring='accuracy')
    mean_accuracy = scores.mean()
    accuracy_scores[name] = mean_accuracy
    print(f'{name}: {mean_accuracy:.3f}')

best_pipeline_name = max(accuracy_scores,key=accuracy_scores.get)
print(f'\nBest pipeline: {best_pipeline_name} (accuracy = {accuracy_scores[best_pipeline_name]:.3f})')
# set variables for test cell
final_pipeline = pipelines_to_try[best_pipeline_name]

# apply final model to test features
# load data

data_test_df = pd.read_csv("test.csv")
# data_test_df.info() # check for additional null values

# median_age = data_test_df['Customer_Age'].median() # note: by default, this will skip NA/null values
# print(f'Median age: {median_age:.2f}')
# data_test_df['Customer_Age'] = data_test_df['Customer_Age'].fillna(median_age)

# convert sex to a 0/1 category
# data_test_df['Gender'].replace(['M','F'],[0,1],inplace=True)

data_test_df = data_test_df[features].astype(float) 

fit = final_pipeline.fit(data_train_ft[features], data_train_tgt)
predictions = fit.predict(data_test_df[features])
'''
def writeSubmission(predictions):
   i=6751
   submissionList = []
   for prediction in predictions:
       submissionList.append([str(i), str(prediction)])
       i+=1
   with open('submission.csv', 'w', newline='') as submission:
       writer = csv.writer(submission)
       writer.writerow(['id', 'Target'])
       for row in submissionList:
           writer.writerow(row)


writeSubmission(predictions)
'''
# This is just using the test.csv to setup a dataframe of the correct size
# and indicies (the "id" field).
make_submission_df = pd.read_csv("test.csv")
# drop all columns except 'id'
make_submission_df = make_submission_df[['id']]
# make sure the column of ID's that we just read in is the index column
make_submission_df = make_submission_df.set_index('id')

# just guess a value from 0 to 5
# probably won't perform very well
predictions = np.random.rand(1350)*5

# Here, you add your predictions to the dataframe
make_submission_df['PTS'] = predictions

# Either one of these will work
# The first one will round all floating point numbers to 2 decimals
# Makes it easier to look at.
make_submission_df.to_csv('submission.csv',sep=',', float_format='%.2f')
#make_submission_df.to_csv('submission.csv',sep=',')