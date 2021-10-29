# Building a time series meta forecasting model with Prophet and LSTM
PyData 2021 Lightning Talk

## Challenge
Mobile telecommunications operators face congestion issues in their networks due to increased usage and environmental factors. As cellular base station congestion negatively impacts customer experience, it can negatively also impact revenue and increase subscriber churn. This notebook demonstrates using time series forecasting to predict network congestion so that the operators are better equipped to manage the situation proactively.

## Why Meta Forecasting?
The first question to address when modelling time series data is how to choose the ‘best model’ among a variety of candidates. Do we adopt statistical methods or other pure machine learning models, including tree-based algorithms or deep learning techniques for Forecasting? Depending on the underlying mechanism of the model and the training data, different models often learn different features, and hence each model can view the data from different perspectives.

In general, statistical techniques are adequate when facing an autoregressive problem (i.e. when the future is related only to the past). At the same time, machine learning and deep learning models are suitable for more complex situations when it’s also possible to combine a large number of data sources. We can achieve higher precision forecasts by combining the power of diverse models, just as in the case of [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) models like random forests.

## Model Selection
In this notebook, we present a technique we call 'meta forecasting' that aims to combine the ability of an additive regression model to learn from experience, along with the generalization and power of deep learning techniques. We build a metamodel using [Prophet](https://facebook.github.io/prophet/) and [long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory). It is similar to [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) but with a slight variation that we'll cover later on.

The reason for selecting [Prophet](https://facebook.github.io/prophet/) is that it provides a stable forecast, and it is designed to deal with country-specific public holidays, missing observations and large outliers. It can also cope with time series that undergo trend changes, such as those due to a product launch, or in the case of the telecoms, when the operator upgrades the infrastructure or changes the cell configuration. In such cases, we can manually input change points to feed in additional information into the model. These effects might not have been well captured by other approaches, including LSTM, making [Prophet](https://facebook.github.io/prophet/) an ideal solution for the first model.

We chose the [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) as the second model due to its ability to forecast for longer time horizons and automatic feature extraction abilities. Furthermore, the different gates inside LSTM boost its capability for capturing nonlinear relationships for forecasting. When modelling time series, there are some factors that have a nonlinear impact on the values we are trying to forecast. Therefore, by using LSTM, the model can learn the nonlinear relationship present in the data leading to better forecast.

## Workflow and Data Processing
1. Fit a [Prophet](https://facebook.github.io/prophet/) model on our training data
2. Extract what [Prophet](https://facebook.github.io/prophet/) has learned and use it to improve the training process of an [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) model

## Model Training
- We fit a [Prophet](https://facebook.github.io/prophet/) model on our raw time series. We add the custom seasonality of the model and try to make its predictions as accurate as possible by changing the Fourier order.
- We now use our fitted Prophet model to improve our [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) training.
- Prophet has now learned the seasonalities present in the data, corrected the anomalous trends, learned the impact of holidays and reconstructed a time series that is devoid of any outliers.
- All these pieces of information are stored in the fitted values. They are a smoothed version of the original data which have been manipulated by the model during the training process. We can view these values as a kind of augmented data source of the original training set.
- Now we start feeding our LSTM [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) with the fitted values produced by [Prophet](https://facebook.github.io/prophet/) and carrying out a multi-step ahead forecast, projecting 148 hours into the future.
- Then we conclude training with the raw data. With our [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), we can also combine external data sources, for example weather conditions, if we think they might have an effect the values we are trying to forecast.

The idea behind such an uncommon approach is that our neural network can learn from two different, but similar, data sources and perform better on our test data. One caveat of this approach is that when performing multi-step training, we have to be mindful of the [catastrophic forgetting problem](https://en.wikipedia.org/wiki/Catastrophic_interference). Catastrophic forgetting is a problem faced by many models and algorithms; when trained on one task, then trained on a second task, many machine learning models “forget” how to perform the first task. To avoid this problem, the structure of the entire network has to be properly tuned to provide a benefit in performance terms.

## Results
At its core, the network is very simple. It is constituted by a [seq2seq](https://en.wikipedia.org/wiki/Seq2seq) LSTM layer that predicts the values of a time series 'n' steps into the future. The training procedure is carried out using [keras-hypetune](https://pypi.org/project/keras-hypetune/).
On our data sets, we made a 13.36% improvement in the RMSE for our meta forecasting model as compared to vanilla LSTM.