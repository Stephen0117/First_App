# This is Final Submission for Miletone_5
### The Apps has been deployed to Heroku (Paas) server and stated as below 
### http://testingapps1234.herokuapp.com/
### Student name: Chong Wei Hong & Chua Cheok Shan
### Matric No: WQD180010 & WQD180096

#### In this submission, we have incorporated a few more enhancement in the project.
#### we have leveraged the time series prediction model (ARIMA) to predict the 10 days rate of change (Case count)
#### of the confirmed covid 19 case. Besides the model has been used for projecting the next 10 days of polarity scores
#### of the tweets. The model was trained using China dataset as it is more initial outbreak country. 
#### The parameters of ARIMA model was optimized by minimizing the AIC value.

#### Ultimately, the visualization is presented in a dashboard using streamlit packages and host it a cloud server (Heroku-Paas)
#### Configurations files has been inputted in the git repository.
#### The project will be further improved in terms of generating a better prediction by accounting more factors in such as split the data into west or east prediction model
#### and taking into account news factors that may/may not affect the progression of the case counts.
