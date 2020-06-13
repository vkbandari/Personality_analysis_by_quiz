# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:14:02 2020

@author: vamshikrishna Bandari
"""
# importing necessary files
import numpy as np
import pandas as pd
import csv	
import matplotlib.pyplot as plt

from flask import Flask, redirect, url_for, request ,render_template
application = Flask(__name__) 

# routing towards first link
@application.route('/')
def Starting():
    return '''<a href = 'http://127.0.0.1:5000/display_qn' > Click on this link to open questions page</a>   '''



# after clicking the link we are requesting to work the below function
@application.route('/display_qn')
def display_qn():
    df =pd.read_csv('datasets/questions.csv')
    
    # passing csv file to web page with column values
    return render_template('index.html', data_length = len(df['s.no']), ques_list = df['question'].to_dict(),\
                           opt1_list = df['option_1'].to_dict(), opt2_list = df['option_2'].to_dict(), \
                               opt3_list = df['option_3'].to_dict(), opt4_list = df['option_4'].to_dict(),\
                                   opt5_list = df['option_5'].to_dict())
    

# creating a list of user values that we got from web page
user_values = []
@application.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
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
      plt.savefig('datasets/result_plot.png')

      # calling the result webpage to show the graph
      return render_template("result.html",result = result)




if __name__ == '__main__':
    application.run(debug=False)
    
    






