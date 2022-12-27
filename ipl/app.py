import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as m

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

#first innings score
filename = 'first-innings-score-lr-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

@app.route('/fi/')
def firstinnings():
	return render_template('firstinnings.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
              
        return render_template('firstinningsresult.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)


## end of first innings score

## match winner prediction

@app.route('/matchwinner',methods=['GET','POST'])
def matchwinner():
  if request.method=='GET':
    return render_template('matchwinner.html')
  if request.method=='POST':
    data=pd.read_csv('cleaned.csv')
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    to_predict = np.array(to_predict_list).reshape(1,-1)
    features=data.iloc[:,[2,4,5,6,7]]
    labels=data.iloc[:,10]
    features_train,features_test,labels_train,labels_test=train_test_split(features,labels,train_size=0.9,random_state=71)
    model=RandomForestClassifier(n_estimators=100,max_depth=25,random_state=129)
    model.fit(features_train,labels_train)
    h=model.predict(to_predict)
    u={1:"Mumbai Indians",2:"Kolkata Knight Riders",3:"Royal Challengers Bangalore",5:"Chennai Super Kings",
                 6:"Rajasthan Royals",7:"Delhi Daredevils",9:"Kings XI Punjab",
                 10:"Sunrisers Hyderabad"}
    return render_template('matchwinner.html',prediction=u[h[0]]+" is having Higher chance to win")


if __name__ == '__main__':
    app.run(debug=True)
