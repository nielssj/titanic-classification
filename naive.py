import os
import pandas as pd

# Load directly into DataFrame
test = pd.read_csv('data/test.csv')
print test

# Naive rule, if you are female you survive
survived = (test['Sex'] == 'female').astype(int)

# Construct result DataFrame
result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
result['Survived'] = survived

# Write to CSV
if not os.path.exists('results'):
    os.makedirs('results')
result.to_csv('results/naive.csv', index=False)
