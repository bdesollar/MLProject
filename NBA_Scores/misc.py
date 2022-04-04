
#print(data_train_ft.isnull().any())

print(data_train_ft.dtypes)

#print(data_train_ft.head(3))

features = ['BIRTHDATE',\
            'SEASON_EXP',
            'HEIGHT',
            'WEIGHT',
            'AST',
            'REB',
            'ALL_STAR_APPEARANCES',
            ]
data_train_ft['BIRTHDATE'] = pd.to_datetime(data_train_ft['BIRTHDATE'])
data_train_ft['BIRTHDATE'] = 2022 - pd.DatetimeIndex(data_train_ft['BIRTHDATE']).year 
data_train_ft['BIRTHDATE'] = data_train_ft['BIRTHDATE'].astype(float)

data_train_ft['SEASON_EXP'] = data_train_ft['SEASON_EXP'].astype(float)

print(data_train_ft['BIRTHDATE'])
pd.set_option('precision', 2)
print(data_train_ft.describe())

data_train_ft['SEASON_EXP'] = data_train_ft['SEASON_EXP'].fillna(0)
data_train_ft['BIRTHDATE'] = data_train_ft['BIRTHDATE'].fillna(0)
data_train_ft['AST'] = data_train_ft['AST'].fillna(0)
data_train_ft['REB'] = data_train_ft['REB'].fillna(0)
data_train_ft['ALL_STAR_APPEARANCES'] = data_train_ft['ALL_STAR_APPEARANCES'].fillna(0)
data_train_ft['HEIGHT'] = data_train_ft['HEIGHT'].fillna(0)
data_train_ft['WEIGHT'] = data_train_ft['WEIGHT'].fillna(0)
data_train_ft = data_train_ft[features]
# data_train_ft['DRAFT_ROUND'].astype(int)


correlation_map = np.corrcoef(data_train_ft.values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()

train_plus_validation_ftrs, test_ftrs, train_plus_validation_tgt, test_tgt = train_test_split (data_train_ft, data_train_tgt, test_size = 0.20,
                                   random_state = 42)

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
for k in range(1,32,2):
    pipelines.append((f'ScaledKNN-{k}', Pipeline([('Scaler', StandardScaler()),(f'KNN-{k}', KNeighborsRegressor(n_neighbors=k))])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, train_plus_validation_ftrs, train_plus_validation_tgt, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


from sklearn.model_selection import GridSearchCV

scaler = StandardScaler().fit(train_plus_validation_ftrs)
rescaledX = scaler.transform(train_plus_validation_ftrs)
param_grid = dict(n_estimators=np.array([50,100,150,200,300,400]))
model = GradientBoostingRegressor(random_state=21)
kfold = KFold(n_splits=10)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, train_plus_validation_tgt)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
from sklearn.metrics import mean_squared_error

scaler = StandardScaler().fit(train_plus_validation_ftrs)
rescaled_train_plus_validation_ftrs = scaler.transform(train_plus_validation_ftrs)
model = GradientBoostingRegressor(random_state=21, n_estimators=400)
model.fit(rescaled_train_plus_validation_ftrs, train_plus_validation_tgt)


# load data
test_df = pd.read_csv("test.csv")
test_df['BIRTHDATE'] = pd.to_datetime(test_df['BIRTHDATE'])
test_df['BIRTHDATE'] = 2022 - pd.DatetimeIndex(test_df['BIRTHDATE']).year 
test_df['BIRTHDATE'] = test_df['BIRTHDATE'].astype(float)

test_df['SEASON_EXP'] = test_df['SEASON_EXP'].astype(float)

print(test_df['BIRTHDATE'])
pd.set_option('precision', 2)
print(test_df.describe())

test_df['SEASON_EXP'] = test_df['SEASON_EXP'].fillna(0)
test_df['BIRTHDATE'] = test_df['BIRTHDATE'].fillna(0)
test_df['AST'] = test_df['AST'].fillna(0)
test_df['REB'] = test_df['REB'].fillna(0)
test_df['ALL_STAR_APPEARANCES'] = data_train_ft['ALL_STAR_APPEARANCES'].fillna(0)
test_df['HEIGHT'] = test_df['HEIGHT'].fillna(0)
test_df['WEIGHT'] = test_df['WEIGHT'].fillna(0)
# test_pts = test_df['PTS']
test_df = test_df[features]

test_df = test_df[features]

pd.set_option('precision', 2)

# transform the validation dataset
rescaled_test_ftrs = scaler.transform(test_df)
predictions = model.predict(rescaled_test_ftrs)

# compare = pd.DataFrame({'Prediction': predictions, 'Test Data' : test_tgt})
# compare.head(10)
# This is just using the test.csv to setup a dataframe of the correct size
# and indicies (the "id" field).
make_submission_df = pd.read_csv("test.csv")
# drop all columns except 'id'
make_submission_df = make_submission_df[['id']]
# make sure the column of ID's that we just read in is the index column
make_submission_df = make_submission_df.set_index('id')

# just guess a value from 0 to 5
# probably won't perform very well
#predictions = np.random.rand(1350)*5

# Here, you add your predictions to the dataframe
make_submission_df['PTS'] = predictions

# Either one of these will work
# The first one will round all floating point numbers to 2 decimals
# Makes it easier to look at.
make_submission_df.to_csv('submission.csv',sep=',', float_format='%.2f')
#make_submission_df.to_csv('submission.csv',sep=',')
