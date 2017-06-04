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
train.reset_index(inplace=True)
train.rename(columns = {'index':'ID'},inplace=True)
print(train.head())


#test = pd.DataFrame.from_dict(test_d, orient='index')
#test.reset_index(level=0, inplace=True)
#test.rename(columns = {'index':'ID'},inplace=True)
#iprint(test.shape)
