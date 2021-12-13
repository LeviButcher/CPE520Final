---
title: Midterm Pump and Dump
author: Levi Butcher
---

<!-- GOAL: 1500 Words -->

- Introduction
    - What is CryptoCurrency
    - What is a Pump And Dump
- Paper Methodology
    - What they Did
    - What Data features they used
    - What models they used
- Paper Results
    - What performance they did reach
- Conclusion / Future Work



# Introduction

CryptoCurrencies are regarded as a digital financial revolution in some areas due to the fact that crypto currency rely on modern technology and are completely unregulated by any goverment. However, This freedom and digital ease of access can lead to some issues with crypto. CryptoCurrencies are a digital currency where all transactions are publicly available and the currency is unregulated by a goverment entity. One of the first cryptocurrencies, Bitcoin, is currently valued at 62,696.50 for a single coin! Other cryptocurrencies are much cheaper such as Dogecoin which is less then a doller currently. Most Crypto Currency are on the cheaper side besides for more mature cryptocurrencies. The cheapness and unregatleness of crypto makes then a perfect target for what's called a Pump and Dump. 

A Pump and Dump is when a group falsely convince other to buy into a stock usually under claims it's a good investment, which causes the the stock price to go up. This phase is the pump. After the pump, the group will then sell all of their own stock causing the stock price to plummet, this is the dump phase. The individuals who invested into this stock then lose all of their investments since the stock price plummets to a value below which they initialize bought at. This scheme is hightly illegal in todays stock market, however, for the unregulated crypto currencies pump and dumps are not illegal. This drives a need for investors to be able to recognize when a pump and dump is happening as quickly as possible and that is what the paper in [1] focuses on.

# Paper Methodology

The paper I focused is [1]. The authors goals where the following:

1. Analyzing different online groups that perform pump and dumps 
2. Creating a publicly available database of pump and dump information
3. Developing their own methodology for detecting if a pump and dump is happening

They gave detailed analysis of how the pump and dump organizes themselves when doing a pump and dump and what forums they communicate on. However, what I'll focus on is there publicly available dataset and their methods for identifying a pump and dump. 

## Data

Their data is publicly availble on Github [2] and contains information on crypto pump and dumps on the biance market place. The dataset contains volume, price, operation type (buy or sell), and the UNIX timestamps for the order. They have records of when a pump and dump took place and download all the Biance market place data from the range of a day before and after the pump and dump start time.

Once they have all the data downloaded they split the data in chunks of size **s** which is in seconds and define a window size **w** in hours. Once the data is split into chunks they aggregate the timeseries information within that chunk of time into the following properties: StdRushOrders, AvgRushOrders, StdTrades, StdVolumes, AvgVolumes, StdPrice, AvgPrice, AvgPriceMax, AvgPriceMin. They go over the method of calculating these properties in [1] but I left it out for brevity. Each aggregate chunk is labeled as a pump and dump depending on if that chunk is within **s** of the pump and dump start time.

Rush Orders is a important feature that they have in their aggregate data, it is defined as being when a users makes several buy orders all at once. This rush order feature they hypothised could be a key indicator that a pump and dump is happening since someone wanting to do a pump and dump would buy a large amount fo the currency all at once.

## Methods

Once the data has ben preprocessed they train two different classification models: Logistic Regression and Random Forest. They based their own work on anomly detection which is why they decided on trying Random Forest. Using the preprocessed aggregated data, it's a pretty straightfoward approach when training. However, their dataset is not as large as they would of liked so they perform 5-fold and 10-fold cross validation to get a more accurate measurement of how their models perform.

# Results

In there results Random Forest performed the best with a F1 score of 91.8% when doing 5-fold CV and having s=25 and w=7. This gives a very fast prediciton of if a pump and dump is happening given the chunk size and window size. They also showed how important each feature in the random forest model and it showed that StdRushOrders was the most important feature when determining if a chunk was part of a pump and dump or not. This proves that Rush orders is a important measurement when identifying pump and dumps.


# Conclusion / Future Work

The paper in [1] gave great perfomance results with simple methods but a clevel approach with using Rush orders. I think their is room for improvement to this approach by utilizing nueral networks. I know Recurrent Neural Networks have been shown to be able to predict stock prices with great success so I hope to apply that method to this dataset to see if I can generate better results then this papers. I also think it would be interesting to be able to incorporate social media data into the feature set since as [1] showed, social media platforms are where these pump and dump groups advertise.
