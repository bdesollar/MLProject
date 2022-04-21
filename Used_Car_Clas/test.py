'''
from sklearn.preprocessing import OneHotEncoder
oneh = OneHotEncoder(handle_unknown="ignore")
data_train_ft = data_train_ft[features]
data_train_ft = oneh.fit_transform(data_train_ft)
# data_train_ft = oneh.transform(data_train_ft)
data_test_df = oneh.transform(data_test_df[features])
data_train_ft[oneh.categories_[0]] = data_train_ft.toarray()
print(data_train_ft.head(10))
# print(data_test_df.head(10))
'''
'''
data_train_ft = pd.get_dummies(data_train_ft[features])
data_test_df = pd.get_dummies(data_test_df[features])
for col in data_train_ft:
    data_train_ft[col] = data_train_ft[col].fillna(0)
    data_test_df[col] = data_test_df[col].fillna(0)

#for col in data_test_df:
    #data_test_df[col] = data_test_df[col].fillna(0)
'''
'''
data_train_ft['year'] = data_train_ft['year'].fillna(0)

data_train_ft['odometer'] = data_train_ft['odometer'].fillna(0)

data_train_ft['lat'] = data_train_ft['lat'].fillna(0)

data_train_ft['long'] = data_train_ft['long'].fillna(0)

data_train_ft['drive'] = data_train_ft['drive'].fillna(0)
data_train_ft['drive'].replace(['4wd','rwd','fwd'],[1, 2, 3],inplace=True)

data_train_ft['size'] = data_train_ft['size'].fillna(0)
data_train_ft['size'].replace(['full-size','compact','mid-size','sub-compact'],[1, 2, 3, 4],inplace=True)

data_train_ft['fuel'] = data_train_ft['size'].fillna(0)
data_train_ft['fuel'].replace(['gas','diesel','other','hybrid','electric'],[1, 2, 3, 4, 5],inplace=True)

data_train_ft['condition'] = data_train_ft['size'].fillna(0)
data_train_ft['condition'].replace(['excellent','fair','like new','good','new','salvage'],[1, 2, 3, 4, 5, 6],inplace=True)

data_train_ft['type'] = data_train_ft['type'].fillna(0)
data_train_ft['type'].replace(['truck','coupe','pickup','SUV','sedan','offroad','hatchback','van',\
 'other','mini-van','wagon','convertible','bus'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],inplace=True)

data_train_ft['title_status'] = data_train_ft['size'].fillna(0)
data_train_ft['title_status'].replace(['clean','rebuilt','salvage','lien','missing','parts only'],[1, 2, 3, 4, 5, 6],inplace=True)


data_train_ft['paint_color'] = data_train_ft['size'].fillna(0)
data_train_ft['paint_color'].replace(['green','white','silver','black','red','yellow','grey','blue'\
 ,'custom','orange','brown','purple'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],inplace=True)

data_train_ft['cylinders'] = data_train_ft['size'].fillna(0)
data_train_ft['cylinders'].replace(['6 cylinders','4 cylinders','8 cylinders','5 cylinders',\
 '10 cylinders','3 cylinders','12 cylinders','other'],[1, 2, 3, 4, 5, 6, 7, 8],inplace=True)


data_train_ft['posting_date'] = pd.to_datetime(data_train_ft['posting_date'], utc=True)
data_train_ft['posting_date'] = pd.to_datetime(data_train_ft['posting_date'])
data_train_ft['posting_date'] = (pd.DatetimeIndex(data_train_ft['posting_date']).day) + (pd.DatetimeIndex(data_train_ft['posting_date']).month*12) + ((2022 - pd.DatetimeIndex(data_train_ft['posting_date']).year)*365)
data_train_ft['posting_date'] = data_train_ft['posting_date'].astype(float)
data_train_ft['posting_date'] = data_train_ft['posting_date'].fillna(0)

#data_train_ft['manufacturer'] = data_train_ft['manufacturer'].fillna(0)
i = 0
for unique_val in data_train_ft['manufacturer'].unique():
    data_train_ft['manufacturer'].replace([unique_val],[i],inplace=True)
    i+=1


data_train_ft = data_train_ft[features]
#print(data_train_ft.dtypes)
#print(data_train_ft.head(5))
for col in data_train_ft:
    print(f'Column ({col}) : {data_train_ft[col].unique()}')
'''