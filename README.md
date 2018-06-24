# Time-Series-Prediction-with-LSTM
We will build a model using some real world internet-of-things (IOT
(https://en.wikipedia.org/wiki/Internet_of_things)) data. As an example we want to predict the daily output of a
solar panel based on the initial readings of the day.

We train the model with historical data of the solar panel. In our example we want to predict the total power
production of the solar panel array for the day starting with the initial readings of the day.

Its an example of many-to-one sequence model.

We will use a recurrent model with the LSTM cell. 

This code has the following sub-sections:

a) Setup

b) Data generation

c) LSTM network modeling

d) Training, test and prediction workflow
