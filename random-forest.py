import os
import pandas as pd
from sklearn import ensemble, cross_validation
from clean import clean_data


''' LOAD '''
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


''' CLEAN '''
clean_data(train)
clean_data(test)
features = ['isFemale', 'Age', 'Parch', 'SibSp', 'Pclass', 'Embarked', 'Fare']


''' TRAIN '''
X = train.as_matrix(features)
Y = train['Survived'].as_matrix()
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit(X, Y)


''' VALIDATE '''
# Validate using 5-fold stratified cross validation
scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


''' TEST '''
result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
result['Survived'] = clf.predict(test.as_matrix(features))


''' OUTPUT '''
if not os.path.exists('results'):
    os.makedirs('results')
result.to_csv('results/random-forest3.csv', index=False)