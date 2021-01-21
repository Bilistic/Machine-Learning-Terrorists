import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import itertools
import numpy as np

"""
takes a labaled column and changes to integer based
"""
def encode(column):
    encoded_content = preprocessing.LabelEncoder()
    return encoded_content.fit_transform(column)

"""
creates a binary array from a column
"""
def binary_array(column):
    return pd.get_dummies(column)


"""
imports a csv file
"""
def import_file(import_path):
    return pd.read_csv(import_path , encoding = "ISO-8859-1", low_memory=False)

"""
writes data frame to file
"""
def export_file(data, path):
    data.to_csv(path, index = True, index_label = 'Id' )

"""
Gets the required data and uses one hot encoding
"""
def preprocess(df):
    year = df.iyear
    month = df.imonth
    day = df.iday
    target_type = binary_array(df.targtype1_txt)
    target_nationality = binary_array(df.natlty1_txt)
    weapon_type = df.weaptype1
    attack_type = binary_array(df.attacktype1_txt)
    country = binary_array(df.country_txt)
    region = binary_array(df.region_txt)
    ans =  pd.concat([year, month, day, target_type, target_nationality, weapon_type, attack_type, country, region], axis=1)
    ans["gname"] = encode(df.gname)
    return ans

"""
Gets the required data and uses label encoding
"""
def preprocessLabel(df):
    temp = pd.DataFrame()
    temp["iyear"] = df.iyear
    temp["imonth"] = df.imonth
    temp["iday"] = df.iday
    temp["targtype1_txt"] = encode(df.targtype1_txt)
    temp["natlty1_txt"] = encode(df.natlty1_txt)
    temp["weaptype1"] = df.weaptype1
    temp["attacktype1_txt"] = encode(df.attacktype1_txt)
    temp["country_txt"] = encode(df.country_txt)
    temp["region_txt"] = encode(df.region_txt)
    temp["gname"] = encode(df.gname)
    features = ["iyear", "imonth", "iday", "targtype1_txt", "natlty1_txt", 
                "weaptype1", "attacktype1_txt", "country_txt", "region_txt"]
    return temp, features

def featureCreate(df):
    features = ["iyear", "imonth", "iday"]
    target_type = set(df.targtype1_txt.values.tolist())
    target_nationality = set(df.natlty1_txt.values.tolist())
    attack_type = set(df.attacktype1_txt.values.tolist())
    country = set(df.country_txt.values.tolist())
    regions = set(df.region_txt.values.tolist())
    features += list(target_type) + list(target_nationality) + list(attack_type) + list(country) + list(regions)
    return features
  
"""
dtree
"""
def dtree(training, validation, features):
    model = tree.DecisionTreeClassifier()
    model = model.fit(training[features], training['gname'])
    classified_pred = model.predict(validation[features])
    return accuracy_score(validation['gname'], classified_pred), classified_pred, validation['gname']

"""
naive bayes
"""
def naiveB(training, validation, features):
    model = BernoulliNB()
    model.fit(training[features], training['gname'])
    classified_pred = model.predict(validation[features]) #gets predictions print("Completed Naive Bayes probabilities")
    return accuracy_score(validation['gname'], classified_pred), classified_pred, validation['gname']

"""
naive bayes
"""
def GnaiveB(training, validation, features):
    model = GaussianNB()
    model.fit(training[features], training['gname'])
    classified_pred = model.predict(validation[features])
    return accuracy_score(validation['gname'], classified_pred), classified_pred, validation['gname']

"""
logistic regression
"""
def logReg(training, validation, features):
    model = LogisticRegression(C=.01)
    model.fit(training[features], training['gname'])
    classified_pred = model.predict(validation[features])
    return accuracy_score(validation['gname'], classified_pred)

"""
K-Neirest Neighbour
"""
def knn(training, validation, features):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(training[features], training['gname'])
    classified_pred = model.predict(validation[features])
    return accuracy_score(validation['gname'], classified_pred), classified_pred, validation['gname']

"""
Random Forrest
"""
def forrest(training, validation, features):
    model = RandomForestClassifier(n_jobs=2, random_state=0)
    model.fit(training[features], training['gname'])
    classified_pred = model.predict(validation[features])
    return accuracy_score(validation['gname'], classified_pred), classified_pred, validation['gname']

"""
Gives a average of each classifiers performance
"""
def BestFitclassifiers(data, target):
    print ("\nThe following are the initial accuracies using CV 5")
    total = []
    model = ["dTree", "nearestN", "nBayes", "randomForest", "GBayes"]
    
    dTree = tree.DecisionTreeClassifier()
    d_scores = model_selection.cross_val_score(dTree, data, target, cv=5)
    total.append(d_scores.mean())
    nearestN = KNeighborsClassifier()
    knn_scores = model_selection.cross_val_score(nearestN, data, target, cv=5)
    total.append(knn_scores.mean())
    nBayes = BernoulliNB()
    nb_scores = model_selection.cross_val_score(nBayes, data, target, cv=5)
    total.append(nb_scores.mean())
    randomForest = RandomForestClassifier()
    rf_scores = model_selection.cross_val_score(randomForest, data, target, cv=5)
    total.append(rf_scores.mean())
    gnb = GaussianNB()
    gnb_scores = model_selection.cross_val_score(gnb, data, target, cv=5)
    total.append(gnb_scores.mean())
    
    print ("Tree : ", total[0])
    print ("NNeighbour : ", total[1])
    print ("Naive Bayes : ",total[2])
    print ("RForest : ",total[3])
    print ("Gausian Naive Bayes : ",total[4])
    p = 0
    for index, x in enumerate(total):
        if p is 0:
            p = index
        elif total[p] < x:
            p = index
    return model[p]

def optimization(model, features, training):
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=3,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           random_state=0, shuffle=False)
    param_grid = { 
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y)
    print(CV_rfc.best_params_)


"""
Based on what was learnt above we choose a classifier and run the test data
"""
def detail_test(model, test, train, features):
    print("Continuing with model: " + model)
    if model is "dTree":
        accuracy, predictions, actual = dtree(train, test, features)
    elif  model is "nearestN":
        accuracy, predictions, actual = knn(train, test, features)
    elif  model is "nBayes":
        accuracy, predictions, actual = naiveB(train, test, features)
    elif  model is "randomForest":
        accuracy, predictions, actual = forrest(train, test, features)
    elif model is "GBayes":
        accuracy, predictions, actual = GnaiveB(train, test, features)
    print("Acuracy against test set = " + str(accuracy))
    cnf_matrix = confusion_matrix(actual, predictions)
    return cnf_matrix
    
"""
Main run operation of script
"""
def main():
    df = import_file("gtd.csv")
    df = df[df.gname != "Unknown"]
    #df = df.groupby("gname").filter(lambda x: len(x) >= 10) #Attempt 1
    df = df.loc[df["gname"].isin(df.groupby("gname").size().nlargest(20).keys())]
    class_names = list(set(df.gname.values.tolist()))
    df["country_txt"] = df.country_txt.str.upper()
    df["natlty1_txt"] = df.natlty1_txt.fillna("unkown")
    #features = featureCreate(df) #Attempt 2 - One hot
    #df = preprocess(df) #Attempt 2 - One hot
    df, features = preprocessLabel(df)
    
    training, validation = train_test_split(df, train_size=.60, random_state=85,
                                            stratify=df["gname"])
    
    model = BestFitclassifiers(training[features], training['gname'])
    matrix = detail_test(model, validation, training, features)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matrix")
    plt.colorbar()
    classes = np.arange(len(class_names))
    plt.xticks(classes, class_names, rotation=45)
    plt.yticks(classes, class_names)
    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('foo.png', bbox_inches='tight',dpi=100)
    plt.show()
    print("Exporting...")
    print("Complete...")
    print("finished")

if __name__ == "__main__":
    main()