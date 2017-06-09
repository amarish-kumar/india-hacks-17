import json
import pandas as pd

training_data_file = "train_data.json"
test_data_file = "test_data.json"
output_file = "output.csv"

with open(training_data_file, 'r') as fptr:
	train_d = json.load(fptr)

#with open(test_data_file, 'r') as fpte:
#	test_d = json.load(fpte)

train = pd.DataFrame.from_dict(train_d, orient='index')
train.reset_index(inplace=True, level=0)
train.rename(columns = {'index':'ID'},inplace=True)
train.replace({"segment":{"neg":"0", "pos":1}})
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
print(dow_sep)
	
	


#test = pd.DataFrame.from_dict(test_d, orient='index')
#test.reset_index(level=0, inplace=True)
#test.rename(columns = {'index':'ID'},inplace=True)
#iprint(test.shape)
