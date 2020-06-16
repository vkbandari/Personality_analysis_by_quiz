# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:14:02 2020

@author: vamshikrishna Bandari
"""
# importing necessary files
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from flask import Flask, url_for, request ,render_template
application = Flask(__name__) 


# routing towards first link
@application.route('/')
def Starting():
    
    return '''<a href = 'http://127.0.0.1:5000/display_qn' > Click on this link to open questions page</a>   '''




# after clicking the link we are requesting to work the below function
@application.route('/display_qn')
def display_qn():
    df =pd.read_csv('static/questions.csv')
    
    # passing csv file to web page with column values
    return render_template('index.html', data_length = len(df['s.no']), ques_list = df['question'].to_dict(),\
                           opt1_list = df['option_1'].to_dict(), opt2_list = df['option_2'].to_dict(), \
                               opt3_list = df['option_3'].to_dict(), opt4_list = df['option_4'].to_dict(),\
                                   opt5_list = df['option_5'].to_dict())
    


# creating a list of user values that we got from web page
user_values = []


# importing preprocessor to use label encoder to process on final label data
from sklearn import preprocessing


# getting object of label encoder
encoder = preprocessing.LabelEncoder()


# to split the data and to shape the data
from sklearn.model_selection import train_test_split


# to check out accuracy uncomment below line
#from sklearn.metrics import accuracy_score


# importing support vector machine algorithm
from sklearn.svm import SVC



@application.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      # getting the data from form
      result = request.form
      for i in result.values():
          user_values.append(float(i))
      
      # if user values is not equal to 40 then uncomment below one
      #user_values = [0.4, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.8,0.4, 0.2, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2,0.4, 0.4, 1, 0.8, 0.8, 0.6, 0.4, 0.8,0.2, 0.4, 0.6, 0.2, 0.8, 0.6, 0.4, 0.8,0.6, 0.4, 0.6, 0.8, 0.8, 1, 0.4, 0.8]
      sum_extr = []
      #print(len(user_values))
      
      # summing of values of each category and finding its percentage
      for i in range(0,len(user_values),8):
          sum_extr.append(round(sum(user_values[i:i+8])/8*100))
      
      #round(sum(user_values
      # initiating the x ticks 
      objects = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'openness', 'Neuroticism']
      print(sum_extr)
      
      # initiating the range of y ticks
      ran = [20,40,60,80,100]
      
      plt.xticks(ran, objects)

      # calling bar plot function
      plt.bar(ran,sum_extr)

      # decorating the graph with axis headings
      plt.ylabel('Percentage')
      plt.xlabel('personality types')

      # making graph title
      plt.title('Percentage obtained on each personality')
      
      plt.show()
      

      # saving graph in datasets folder
      plt.savefig('static/result_plot.png')
      
      # model deployment to predict the personality
      
      # reading the dataset which is in csv format
      df = pd.read_csv('static/model_dataset.csv')
      
      # training the label data to encode
      encoder.fit(df.Type)
      
      # conveting all label data in df[Type] to numeric
      df['Type'] = encoder.transform(df['Type'])
      
      # taking the rest columns to x
      x = df[df.columns[:-1]]
      
      # taking last column to y
      y = df['Type'] 
      


      '''
      # randomly splitting and arranging the data
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
      
      # initiated the svm model
      model_1 = SVC()
      
      # training the model by passing input and output
      model_1.fit(x_train,y_train)
      
      # predicting the output by passing only input data
      y_pred_svm = model_1.predict(x_test)
      
      # checking accuracy by passing true output and predicted output      
      print(accuracy_score(y_test, y_pred_svm)*100)
      
      #printing the value
      for i,j in zip(y_test,y_pred_svm):
          print('actual:\t%s\t\tpredicted:\t%s'%(i,j))
      
      '''


      
      # randomly splitting and arranging the data
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, random_state=42)
      
      # initiated the svm model
      model= SVC()
      
      # training the model by passing input and output
      model.fit(x_train,y_train)
      
      # taking the labeles that encoder encoded
      labels = encoder.classes_
      
      #print(labels)
      
      # here the actual prediction takes place by passing all percentages of the answers the user given
      label_predicted_value = model.predict([sum_extr])
      
      # based on predicted number we are getting title of predicted
      label_name = labels[label_predicted_value]


      
      # to get printed of predicted class and percentages uncomment the below two statements
      #print(label_name[0])
      #print(sum_extr)
      

      
      # calling the result webpage to show the graph
      return render_template("result.html",label_name =label_name[0])




if __name__ == '__main__':
    application.run(debug=False)
    
    





