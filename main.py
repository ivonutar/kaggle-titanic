import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Read data

train_dataset = pd.read_csv("data/train.csv", index_col=0)
test_dataset = pd.read_csv("data/test.csv")

# Concat dataframes for feature engineering
train_dataset_len = len(train_dataset)
whole_dataset = pd.concat([train_dataset, test_dataset], sort=False)

# TODO Choose features
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
features = 'Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'
features = features.split(',')

# TODO fill NaNs
# Fill Age
# Add median, if no title, otherwise ??? older person with title
age_filler = whole_dataset['Age'].median()
whole_dataset['Age'] = whole_dataset['Age'].fillna(age_filler)

# File SibSp
SibSp_filler = whole_dataset['SibSp'].median()
whole_dataset['SibSp'] = whole_dataset['SibSp'].fillna(SibSp_filler)

# File Parch
SibSp_filler = whole_dataset['Parch'].median()
whole_dataset['Parch'] = whole_dataset['Parch'].fillna(SibSp_filler)

# Fill ticket
whole_dataset['Ticket'] = whole_dataset['Ticket'].fillna('0')

# Fill Fare
SibSp_filler = whole_dataset['Fare'].mean()
whole_dataset['Fare'] = whole_dataset['Fare'].fillna(0)

# Fill Cabin
# A,B,C,D,E + number..., fill NaN with X0
whole_dataset['Cabin'] = whole_dataset['Cabin'].fillna('X0')

# Fill Embarked
# whole_dataset['Embarked'] = whole_dataset['Embarked'].median()

# TODO engineer features

# Fare categories
label = LabelEncoder()
whole_dataset['FareBin'] = pd.qcut(whole_dataset['Fare'], 4)
whole_dataset['FareBin_Code'] = label.fit_transform(whole_dataset['FareBin'])

# Age categories
whole_dataset['AgeBin'] = pd.qcut(whole_dataset['Age'], 4)
whole_dataset['AgeBin_Code'] = label.fit_transform(whole_dataset['AgeBin'])

# Title
whole_dataset['Title'] = whole_dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
whole_dataset['Title_Code'] = label.fit_transform(whole_dataset['Title'])


# Append features
features.append('FareBin_Code')
features.append('AgeBin_Code')
features.append('Title_Code')


# TODO convert string features
whole_dataset['Sex_Code'] = label.fit_transform(whole_dataset['Sex'])
whole_dataset["Embarked"] = whole_dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2})
whole_dataset['Embarked_Code'] = label.fit_transform(whole_dataset['Embarked'])

whole_dataset['IsChild'] = [1 if i < 16 else 0 for i in whole_dataset.Age]

features = ['FareBin_Code', 'AgeBin_Code', 'Title_Code', 'Sex_Code', 'Embarked_Code', 'IsChild', 'Pclass']


# TODO split data

train_dataset = whole_dataset[:train_dataset_len]
test_dataset = whole_dataset[train_dataset_len:]

X_train = train_dataset.drop("Survived", axis=1)[features]
Y_train = train_dataset["Survived"]
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=101)


# TODO Choose model
random_state = 2
classifiers = [SVC(random_state=random_state), DecisionTreeClassifier(random_state=random_state),
               AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                  learning_rate=0.1), RandomForestClassifier(random_state=random_state),
               ExtraTreesClassifier(random_state=random_state), GradientBoostingClassifier(random_state=random_state),
               MLPClassifier(random_state=random_state), KNeighborsClassifier(),
               LogisticRegression(random_state=random_state), LinearDiscriminantAnalysis()]

cv_results = []
kfold = StratifiedKFold(n_splits=10)

for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# TODO Get index
classifier_index = cv_means.index(max(cv_means))

# TODO Fit model

y = train_dataset.Survived
X = train_dataset[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# SVC
model = classifiers[classifier_index]
model.fit(train_X, train_y)


# TODO Evaluate (crossval)

crossval = cv_means[classifier_index]
print("Cross validation score: {}".format(crossval))

# TODO Predict
X_test = test_dataset[features]

PassengerId = test_dataset['PassengerId']
Submission = pd.DataFrame()
Submission['PassengerId'] = test_dataset['PassengerId'].astype(int)

Submission['Survived'] = model.predict(X_test)
Submission['Survived'] = Submission['Survived'].astype(int)
Submission.to_csv('predictions-main.csv', index=False)
