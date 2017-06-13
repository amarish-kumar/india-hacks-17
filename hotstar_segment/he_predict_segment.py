from sklearn.ensemble import RandomForestClassifier
import pandas as pd

training_data_file = "train_data.json"
test_data_file = "test_data.json"
output_file = "output.csv"

train = pd.read_json(training_data_file,orient="index")
train.reset_index(inplace=True, level=0)
train.loc[train['segment'] == 'neg', 'segment'] = "0"
train.loc[train['segment'] == 'pos', 'segment'] = "1"
print(train.head())
#d = train["dow"]
#print(d)
#print(type(d))
#train.loc[:, "dow1"] = pd.Series([{'some shit':1, "some other shit":4}]*200000, index=train.index)
#print(train["dow1"])

dow_sep = []
for x in train["dow"]:
	ds = {d.split(':')[0]:d.split(':')[1] for d in x.split(',')}
	dow_sep.append(ds)
#print(dow_sep)

target_features = train['segment'].tolist()
#print(target_features)

classifier = RandomForestClassifier(n_estimators=500,max_depth=12, max_features=10)
classifier.fit(dow_sep, target_features)



#test = pd.DataFrame.from_dict(test_d, orient='index')
#test.reset_index(level=0, inplace=True)
#test.rename(columns = {'index':'ID'},inplace=True)
#iprint(test.shape)
