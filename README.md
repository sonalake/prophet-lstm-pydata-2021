# # PyData 2021 Lightning Talk: _Building a Meta Forecasting Model with Prophet and LSTM for Time series Forecasting_

#### Description of the Problem and the Data:
Mobile operators face congestion problems in their networks due to increased usage and other factors. Congested sites (cellular base stations) will lose revenue
because usage at these sites is curtailed. In this notebook, we use time series forecasting to predict network congestion so that the companies can be better equipped to handle the network traffic.

### 1. Need for a Meta Forecasting Model

- First question that we come across while modelling time series data is how to choose the ‘best model’ among a variety of candidates. Whether to adopt statistical methods or other pure machine learning models, including tree-based algorithms or deep learning techniques for Forecasting 
- Depending on the underlying mechanism of the model and the training data, different models often learn different features and hence each model can explain different perspectives of data
- In general: statistical techniques are adequate when we face an autoregressive problem i.e. when the future is related only to the past; while machine learning and deep learning models are suitable for more complex situations when it’s also possible to combine a large number of data sources
- Higher precision Forecasts can be achieved if we combine the power of Diverse Models just as in case of Ensemble learning models such as Random Forests

### 2. Model Selection 

- In this notebook, we present a technique which we call 'Meta Forecasting' that tries to combine the ability of an additive regression model to learn from experience with the generalization and power of deep learning techniques 
- We build a Meta Model using Prophet and LSTM. It is similar to Ensemble learning but with a slight variation which we will understand as we proceed further.
- The reason behind this conscious decision to select Prophet is due to the fact that it provides a stable forecast and the Prophet is designed to deal with country-specific public holidays, missing observations and large outliers. It is also designed to cope with time series that undergo trend changes, such as a product launch, or in the case of the telecoms, when the company upgrades the infrastructure or changes the cell configuration we can manually input the changepointd to feed in additional information into the model. These effects might not have been well captured by other approaches including LSTM, making Prophet an ideal solution for the first model
- We chose the Long Short-Term Memory (LSTM) as the second model due to its ability to forecast for longer time horizons and automatic feature extraction abilities. Furthermore, the different gates inside LSTM boost its capability for capturing non-linear relationships for forecasting. While modelling time series there are some factors that have a non-linear impact on demand and therefore by using LSTM the model can learn the nonlinear relationship present in the data leading to better forecast. 


### 3. Workflow and Data Processing 

- Fit a Prophet Model on our training data;
- Extract what Prophet has learned and use it to improve the training process of an LSTM model performing a two-step training


### 4. Training the Models:

- We fit a Prophet model on the Raw Time series. We add the custom seasonality of the model and try to make its predictions as accurate as possible
- Now our scope is to use our fitted Prophet model to improve the training of our neural network. Prophet has learned the multiple seasonalities present in the data, corrected the anomalous trends, learned the impact of the holidays and reconstructing a time series that is devoid of any outliers
- All these pieces of information are stored in the fitted values, they are a smoothed version of the original data which have been manipulated by the model during the training procedure. In other words, we can see these values as a kind of augmented data source of the original train set
- Our strategy involves applying a two-step training procedure. We start feeding our LSTM autoencoder, using the fitted values produced by Prophet, for multi-step ahead forecasts, projecting 148 hours into the future.  - Then we conclude the training with the raw data, in our case they are the same data we used before to fit the Prophet. With our neural network, we can also combine external data sources, for example, the weather conditions if we think those set of external parameters might affect the KPI
- The idea behind such uncommon approach is that our neural network can learn from two different but similar data sources and perform better on our test data.
- one Caveat of this approach is that When performing multiple-step training we have to take care of the *Catastrophic Forgetting problem*. Catastrophic forgetting is a problem faced by many models and algorithms. When trained on one task, then trained on a second task, many machine learning models “forget” how to perform the first task
- To avoid this tedious problem, the structure of the entire network has to be properly tuned to provide a benefit in performance terms. From these observations, we preserve a final part of our previous training as validation
- At its core, the network is very simple. It’s constituted by a seq2seq LSTM layer which predicts the values of the KPI N steps ahead in the future. The training procedure is carried out using keras-hypetune. This framework provides hyperparameter optimization of the neural network structures in a very intuitive way.


### Results

Obtained 13.36% improvements in the RMSE for Meta Forecasting Model Compared to Vanilla LSTM model.