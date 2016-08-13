import os
import pandas as pd
import pydot
from sklearn import tree, cross_validation
from sklearn.externals.six import StringIO
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
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
clf.score(X, Y)


''' VALIDATE '''
# Validate using 5-fold stratified cross validation
scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


''' VISUALIZE '''
# Export dot visualization
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('graph.pdf')


''' TEST '''
result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
result['Survived'] = clf.predict(test.as_matrix(features))


''' OUTPUT '''
if not os.path.exists('results'):
    os.makedirs('results')
result.to_csv('results/decision-tree2.csv', index=False)