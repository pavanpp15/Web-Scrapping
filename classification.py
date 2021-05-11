import pandas as pd
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text



def build_test_model(df , df_test):
    
    #1 PreProcess the training data
     
    df['JOB DESCRIPTION'] = df['JOB DESCRIPTION'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df['JOB DESCRIPTION'] = df['JOB DESCRIPTION'].str.lower().str.split()
    stop = stopwords.words('english')
    df['JOB DESCRIPTION'] = df['JOB DESCRIPTION'].apply(lambda x: [item for item in x if item not in stop])
    df['JOB DESCRIPTION']= df['JOB DESCRIPTION'].str.join(" ")
    df['JOB DESCRIPTION'] = df['JOB DESCRIPTION'].apply(remove_punctuations)
    df['JOB DESCRIPTION'] = df['JOB DESCRIPTION'].str.replace('\d+', '')

    df['JOB TITLE'] = df['JOB TITLE'].str.lower()
    df['JOB DESCRIPTION'] = [re.sub('data eng[a-z]+', '',x) for x in df['JOB DESCRIPTION']]
    df['JOB DESCRIPTION'] = [re.sub('data sci[a-z]+', '', x) for x in df['JOB DESCRIPTION']]
    df['JOB DESCRIPTION'] = [re.sub('software eng[a-z]+', '', x) for x in df['JOB DESCRIPTION']]

    # Preporcess the Test File
    df_test.iloc[:,0] = df_test.iloc[:,0].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_test.iloc[:,0] = df_test.iloc[:,0].str.lower().str.split()
    df_test.iloc[:,0] = df_test.iloc[:,0].apply(lambda x: [item for item in x if item not in stop])
    df_test.iloc[:,0]= df_test.iloc[:,0].str.join(" ") 
    df_test.iloc[:,0] = df_test.iloc[:,0].apply(remove_punctuations)
    df_test.iloc[:,0] = df_test.iloc[:,0].str.replace('\d+', '')
    

    
    df_test.iloc[:,0] = [re.sub('data eng[a-z]+', '',x) for x in df_test.iloc[:,0]]
    df_test.iloc[:,0] = [re.sub('data sci[a-z]+', '', x) for x in df_test.iloc[:,0]]
    df_test.iloc[:,0] = [re.sub('software eng[a-z]+', '', x) for x in df_test.iloc[:,0]]

    # Vectorize the test and train data

    jobs_desc = df['JOB DESCRIPTION']
    jobs_desc_test = df_test.iloc[:,0]
    counter = CountVectorizer(ngram_range=(1,2))
    
    counter.fit(jobs_desc)
    counts_train = counter.transform(jobs_desc)#transform the training data
    counts_test = counter.transform(jobs_desc_test)

    # Transform
    transformer = TfidfTransformer()
    counts_train_tfidf = transformer.fit_transform(counts_train)
    counts_test_tfidf = transformer.fit_transform(counts_test)

    Y = df['JOB TITLE']

    le = preprocessing.LabelEncoder()
    y_train=le.fit_transform(Y)
   

    # Models

    #1. SVC Normal
    clf_svc = LinearSVC()
    clf_svc.fit(counts_train_tfidf,y_train)
    pred_Linear_Svc=clf_svc.predict(counts_test_tfidf)

    #2. SVC Grid
    # We tried Grid Search SVC and got the accuracy score of 83.95
    """
    clf_grid = LinearSVC()
    parameters = {'penalty':['l2'], 'loss':['hinge', 'squared_hinge'],'multi_class':['ovr', 'crammer_singer'],'C':[.5,.75,1,1.5,2],'class_weight':['balanced','None'],'max_iter':[2000],'verbose':[1], 'random_state':[7,91,173]}
    grid = GridSearchCV(clf_grid, parameters, cv=3)
    grid.fit(counts_train_tfidf,y_train)
    pred_svc_grid=grid.predict(counts_test_tfidf)
    """
    
    #3. Logistic Reg
    # We tried Logistic Reg and got the accuracy score of 78.95
    """
    logistic_regression = LogisticRegression(multi_class = "multinomial",solver = 'sag',warm_start=True,max_iter=5000,verbose=1)
    logistic_regression.fit(counts_train_tfidf,y_train)
    y_pred_lreg = logistic_regression.predict(counts_test_tfidf)
    """

    #4. Grid Logistic reg
    # We tried Grid search Logistic Reg and got the accuracy score of 83.95
    """
    LREG_grid = [ {'multi_class':['auto'],'C':[0.1,1,10,100],'penalty':['l2'],'tol':[0.0001,0.01,0.1], 'solver':['sag'],'random_state':[5,20,10,15],'max_iter':[5000], 'warm_start':['True']}] 
    LREG_classifier=LogisticRegression()
    # build a grid search to find the best parameters
    gridsearchLREG  = GridSearchCV(LREG_classifier, LREG_grid, cv=5)
    lreg_fit = gridsearchLREG.fit(counts_train_tfidf,y_train)
    predicted=gridsearchLREG.predict(counts_test_tfidf)
    """
    
    #5. Decision Tree
    # We tried Decision Tree and got the accuracy score of 75.8
    """
    dt = DecisionTreeClassifier()
    dt.fit(counts_train_tfidf,y_train)
    y_pred=dt.predict(counts_test_tfidf)
    from sklearn.metrics import accuracy_score
    print (accuracy_score(y_pred,y_test))
    """
    
    submission = pd.DataFrame(le.inverse_transform(pred_Linear_Svc))
    submission.to_csv("Jobs_pred.csv",index=False, header=False)    
    print("File pred saved")

def predict_test_input(path):
    try:
        #Read test and train data
        df_test = pd.read_csv(path, header=None) 
        df = pd.read_csv('./Train.csv')
        build_test_model(df, df_test)
    except:
        print("Can not read the test input file at this path")

path_to_csv = './Test.csv'       # Please keep the test file in the same folder as this file 
predict_test_input(path_to_csv)