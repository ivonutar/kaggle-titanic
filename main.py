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


# TODO Choose features
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
features = 'Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'
features = features.split(',')

# TODO fill NaNs
# Fill Age
# Add median, if no title, otherwise ??? older person with title
age_filler = train_dataset['Age'].median()
train_dataset['Age'] = train_dataset['Age'].fillna(age_filler)

# File SibSp
SibSp_filler = train_dataset['SibSp'].median()
train_dataset['SibSp'] = train_dataset['SibSp'].fillna(SibSp_filler)

# File Parch
SibSp_filler = train_dataset['Parch'].median()
train_dataset['Parch'] = train_dataset['Parch'].fillna(SibSp_filler)

# Fill ticket
train_dataset['Ticket'] = train_dataset['Ticket'].fillna('0')

# Fill Fare
SibSp_filler = train_dataset['Fare'].mean()
train_dataset['Fare'] = train_dataset['Fare'].fillna('0')

# Fill Cabin
# A,B,C,D,E + number..., fill NaN with X0
train_dataset['Cabin'] = train_dataset['Cabin'].fillna('X0')

# Fill Embarked
# train_dataset['Embarked'] = train_dataset['Embarked'].median()

# TODO engineer features

# Fare categories
label = LabelEncoder()
train_dataset['FareBin'] = pd.qcut(train_dataset['Fare'], 4)
train_dataset['FareBin_Code'] = label.fit_transform(train_dataset['FareBin'])

# Age categories
train_dataset['AgeBin'] = pd.qcut(train_dataset['Age'], 4)
train_dataset['AgeBin_Code'] = label.fit_transform(train_dataset['AgeBin'])

# Title
train_dataset['Title'] = train_dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train_dataset['Title_Code'] = label.fit_transform(train_dataset['Title'])


# Append features
features.append('FareBin_Code')
features.append('AgeBin_Code')
features.append('Title_Code')


# TODO convert string features
train_dataset['Sex_Code'] = label.fit_transform(train_dataset['Sex'])
train_dataset["Embarked"] = train_dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2})
train_dataset['Embarked_Code'] = label.fit_transform(train_dataset['Embarked'])

train_dataset['IsChild'] = [1 if i<16 else 0 for i in train_dataset.Age]

features = ['FareBin_Code', 'AgeBin_Code', 'Title_Code', 'Sex_Code', 'Embarked_Code', 'IsChild', 'Pclass']


# TODO split data

X_train = train_dataset.drop("Survived", axis=1)[features]
Y_train = train_dataset["Survived"]
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=101)


# TODO Choose model
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
kfold = StratifiedKFold(n_splits=10)

for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
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

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# SVC
model = classifiers[classifier_index]
model.fit(train_X, train_y)


# TODO Evaluate (crossval)

crossval = cv_means[classifier_index]
print("Cross validation score: {}".format(crossval))

# TODO Predict