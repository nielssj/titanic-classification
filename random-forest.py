import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from clean import clean_data


''' LOAD '''
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


''' CLEAN '''
clean_data(train)
clean_data(test)
features = ['isFemale', 'Age', 'Parch', 'SibSp', 'Pclass', 'Embarked', 'Fare']
X = train.as_matrix(features)
Y = train['Survived'].as_matrix()


''' TUNE CLASSIFIER '''
# Initialize different classifiers
clf = RandomForestClassifier(max_features=None, oob_score=True)
error_rate = []

# Range of parameter values to explore.
min_param = 2
max_param = 20
params = range(min_param, max_param + 1)

# Run classifiers with increasing max_depth
for i in params:
    clf.set_params(n_estimators=50, max_depth=i)
    clf.fit(X, Y)
    oob_error = 1 - clf.oob_score_
    error_rate.append((i, oob_error))

# Plot result
xs, ys = zip(*error_rate)
plt.plot(xs, ys)
plt.xlim(min_param, max_param)
plt.xlabel("max_depth")
plt.ylabel("OOB error rate")
plt.show()


''' TRAIN '''
clf.set_params(n_estimators=50, max_depth=10)
clf.fit(X, Y)


''' TEST '''
result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
result['Survived'] = clf.predict(test.as_matrix(features))


''' OUTPUT '''
if not os.path.exists('results'):
    os.makedirs('results')
result.to_csv('results/random-forest4.csv', index=False)
