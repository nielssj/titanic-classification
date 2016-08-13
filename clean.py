import numpy as np


def clean_data(frame):
    # Set missing age fields to average (TODO: Use other features such as name to make more intelligent guess)
    meanAge = frame[frame['Age'].isnull() == False]['Age'].median()
    frame['Age'] = frame['Age'].fillna(meanAge)

    # Convert gender to boolean
    genders = {'female': 0, 'male': 1}
    frame['isFemale'] = frame['Sex'].map(lambda x: genders[x])

    # Set missing embarked to most common
    frame['Embarked'][frame['Embarked'].isnull()] = frame['Embarked'].dropna().mode().values

    # Convert embarked from string to int
    ports = {'C': 0, 'Q': 1, 'S': 2}
    frame['Embarked'] = frame['Embarked'].map(lambda x: ports[x])

    # Set missing fares to median of class
    if len(frame['Fare'][frame['Fare'].isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0, 3):  # loop 0 to 2
            median_fare[f] = frame[frame['Pclass'] == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            frame.loc[(frame['Fare'].isnull()) & (frame['Pclass'] == f + 1), 'Fare'] = median_fare[f]
