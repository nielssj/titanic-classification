import csv
# noinspection PyUnresolvedReferences
from pyspark.sql.types import *
# noinspection PyUnresolvedReferences
from pyspark.sql import SparkSession
# noinspection PyUnresolvedReferences
from pyspark.mllib.regression import LabeledPoint
# noinspection PyUnresolvedReferences
from pyspark.mllib.tree import RandomForest

spark = SparkSession\
    .builder\
    .appName("RandomForest")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()


''' LOAD '''
train = spark.read.csv('data/train.csv', header=True, inferSchema=True)
test = spark.read.csv('data/test.csv', header=True, inferSchema=True)


''' CLEAN '''
medianAge = train \
            .where(train['Age'] != None) \
            .approxQuantile('Age', [0.5], 0.25)[0]

def clean_data(passenger):
    clean_row = {}
    # Set missing age fields to average (TODO: Use other features such as name to make more intelligent guess)
    if passenger['Age'] is None:
        clean_row['Age'] = medianAge
    else:
        clean_row['Age'] = float(passenger['Age'])
    if passenger['Sex'] == 'female':
        clean_row['isFemale'] = 1
    else:
        clean_row['isFemale'] = 0
    if 'Survived' in passenger:
        clean_row['Survived'] = passenger['Survived']
    return clean_row

train2 = train.rdd.map(clean_data)
test2 = test.rdd.map(clean_data)


''' TRAIN '''
train2 = train2.map(lambda p: LabeledPoint(p['Survived'], [p['Age'], p['isFemale']]))
model = RandomForest.trainClassifier(train2, 2, {}, 3, seed=42)


''' TEST '''
test2 = test2.map(lambda p: [p['Age'], p['isFemale']])
result = model.predict(test2)


''' OUTPUT '''
output = test.rdd.zip(result)
output = output.map(lambda r: {'PassengerId': r[0]['PassengerId'], 'Survived': int(r[1])})
with open('results/spark-random-forest.csv', 'wb') as csv_file:
    writer = csv.DictWriter(csv_file, ['PassengerId', 'Survived'])
    writer.writeheader()
    for row in output.collect():
        writer.writerow(row)
