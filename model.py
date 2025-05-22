import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report
import pickle as pickle
import joblib

def create_model(data):
    x = data.drop(columns = ["heartdisease"],axis=1) 
    y=data["heartdisease"]
    
    feature_columns = x.columns.tolist()
    #splitting the data
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = .2, random_state = 42 )
    
    #train the data
    model = LogisticRegression(max_iter=1000) 
    model.fit(x_train,y_train)

    #test model
    
    #fit to predict
    y_pred = model.predict(x_test)
    print("Accuracy of our model: ", accuracy_score(y_test,y_pred))
    print("Classification report: \n", classification_report(y_test,y_pred))
    #accuracy is 0.8532608695652174
    #accuracy of 89% for heart attack & 80% for no hear
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return model,feature_columns
    
def get_clean_data(): #reading the file

    data = pd.read_csv("C:/Users/Telka LLC/Desktop/Jupytr/streamlitapp_heart_rate_pred/heart.csv")
    data.columns = data.columns.str.lower()
    #creating dummy variables:
    data = pd.get_dummies(data, drop_first=True)
    
    data = data.drop(columns=['restingbp', 'chestpaintype_ta', 'restingecg_normal'], errors='ignore')
    data = data.drop(columns=['restingbp', 'chestpaintype_ta', 'restingecg_normal'], errors='ignore')


    
    return data



def main(): 
    data = get_clean_data() #only works when I want it to
    print(data.head())
    print(data.info())
    
    model, feature_columns = create_model(data)

    joblib.dump(model, "model.pkl")
    joblib.dump(feature_columns, "columns.pkl") 
    
   
if __name__ =="__main__":
    main()
#cholesterol,age,maxhr,chest pain type,exercise angina, stslope flat & up are more related


#print(data.columns.tolist())






