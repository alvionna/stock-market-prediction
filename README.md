# Stock Analysis & Prediction for 4 Giant Techs

## Overview

In this project, I aim to do a stock analysis for 4 giant tech companies: Apple, Amazon, Google, and Microsoft. Once the analysis done, I will build a Machine Learning/Deep Learning model to predict the closing price movement in the future.

NOTE: all of the observations were made on December 22nd, 2023. These observations will no longer be valid for the periods after this date.

### Data Sources

This project will use the data taken from yahoo finance by importing `yfinance` that provides real time relevant stock data, such as closing price, volume sales, etc.

### Project Structure

There are two folders in this repository: `images` and `src`.

- [images](https://github.com/alvionna/stock-market-prediction/blob/main/images) contains the images that I displayed in this README file.
  - To see the complete graphs of this project, please go to `main.ipynb` in the [src](https://github.com/alvionna/stock-market-prediction/blob/main/src) file.
- [src](https://github.com/alvionna/stock-market-prediction/blob/main/src) contains the coding files for this project.
  - There is only one file in this folder: `main.ipynb`. This file is the heart of this project.

## Learning Focus

1. Practice visualization utilizing `matplotlib` and `seaborn`
2. Learn more about stock technical analysis to predict future price movements based on historical data
3. Getting started with risk management
4. Learn more about the application of ML/DL in finance.

## My Progress

1. Done with the basic analysis of the stocks
   - Except for the risk management
2. Done with the basic model to predict future price movement using LSTM

## Next Steps

1. Use _StandardScaler_ rather than _MinMaxScaler_ since stock prices are ever-changing, and there are no true min or max values. While it may not lead to a disastrous event by using _MinMaxScaler_, it doesn't make sense to use _MinMaxScaler_.

## Experiment & Result

### Data Analysis

As of December 20th, 2023, there is 1004 rows in the dataset. This means that the dataset does not record any data happening in the weekends.
![df_describe](https://github.com/alvionna/stock-market-prediction/blob/main/images/df_describe.png)

### Stock Analysis

In this analysis, I tried to answer several questions related to stock analysis:

1. [What was the change in price of the stocks overtime?](####what-was-the-change-in-price-of-the-stocks-overtime?)
2. [What was the moving average of the various stocks?](<####what-was-the-moving-average-of-the-various-stocks?-(What-are-the-trends?)>)
3. [What was the average daily return of the stock?](####what-was-the-average-daily-return-of-the-stock?)
4. [What was the correlation between different stocks' closing prices?](####what-was-the-correlation-between-different-stocks'-closing-prices?)
5. [How much value do we put at risk by investing a particular stock?](####how-much-value-do-we-put-at-risk-by-investing-a-particular-stock?)

#### What was the change in price of the stocks overtime?

- To answer this question, I plotted a graph of the daily closing price and the sales volume. In addition, I created a histogram of the sales volume to clearly see which company has the highest sales volume.
- These are some observations made:
  - All of the companies showed that their stock has matured.
  - Microsoft has the highest growth rate compared to the other companies (more fundamental analysis is required)
  - The question becomes "Why does Microsoft have the highest growth rate when their stock has matured?'
- From the graph and histogram of daily sales volume, these are some observations made:
  - It can be seen from the graph that Apple and Amazon have higher trades than Microsoft and Google.

#### What was the moving average of the various stocks? (What are the trends?)

- To answer this question, I did calculations over three time periods, [20, 50, 100] days, of the moving average to **observe the pattern or trend** of the stock.

* These are the observations made (as of December 22nd, 2023):
  - For 20 Days:
    - Apple, Google, and Amazon are currently on an uptrend as the closing price > moving average. However, Apple needs to watch out for a potential reversal since the closing price is currently going down.
    - Microsoft is currently on a downtrend since the closing price < moving average.
  - For 100 Days:
    - A little bit different from observations made when we our look-back period was 20 days, the moving average with a look-back period of 100 days indicates that ALL comapnies are having an uptrend.
    - Mircosoft and Apple (again) need to watch out since their stock prices are decreasing.
  - Plotting the short-term (20 days) and long-term (100 days) moving average simultaneously:
    - All the companies have a tiny dip a little bit before and around November 2023 before their closing price rises. This indicate there might be a major events during those moments that impact the closing price of these companies.
    - Apple's stock's graph indicate a buy signal or provide a golden cross around the end of 2023-11 or early 2023-12.
    - Google's stock is having an uptrend in general
    - Microsoft's stock's graph indicate a death cross around 2023-10 and golden cross on 2023-11. However, as the moving average is above the proce, this could signal a potential reversal and it might not be prudent to buy for a short-term investment.
    - Amazon's stock's graph indicate a death cross around 2023-10 and golden cross around 2023-11. Currently, Amazon is observing an uptrend in general.

- However, moving average is calculated from historical data and it is **nothing predictive in nature**.
  - Results using moving averages can be random and are more prone to giving _false signals_.
  - While the market respects the signals observed from moving average at times, it can be deduced that better indicator to clarify the trend of the stocks.

#### What was the average daily return of the stock?

- To answer this question, I plotted the daily return according to their dates to see the spikes and dips of the daily return. Moreover, I created histograms and calculated the standard devation to see the spread and **volatility** of the stocks.
- These are the observations made (as of December 22nd, 2023):

* Apple has the most consistent daily return since there's relativly no apparent points that are particularly high or low.
  - Moreover, Apple possesses the smallest standard deviation (0.013) and spread as can be seen from the histogram
  - Google and Amazon have more volatile daily returns with more spikes and dips compared to the rest.
    - This observation can be confirmed by looking at
      - The standard deviation where the standard deviation of the daily return of Google's and Amazon's stock (0.019 and 0.021, respectively) is higher than the other two stocks.
      - The histogram where the spread of daily return of Google's and Amazon's stocks are wider than the other two stocks.

#### What was the correlation between different stocks' closing prices?

- To answer this answer, I created a heatmap with numerical values to provide a clearer visualization of the correlation between companies.
- These are the observations made:
  - Google and Amazon have the strongest correlation on closing prices.
  - Google and Apple have the weakest correlation on closing prices.

#### How much value do we put at risk by investing a particular stock?

- Investing in Amazon's stock poses the highest risk among the companies. However, it is also expected that Amazon's stock has the highest return among the 4 giant techs.
  - The safest stock to invest in is the Apple's stock as it poses the lowest risk. However, Apple's stock also provides the lowest return among the companies.
  - It's interesting to see that, while investing in Google poses a higher risk, an investor may gain a lower return compared to investing in Microsoft.

### Stock Prediction

- We build a model to predict the future stock price for each company based on historical data, where we started from 2012.

* We utilized the _closing price_ as our variable to predict the future stock price.

- We utilized _Bi-LSTM_ to predict the price movements and utilized _RMSE_ as the model evaluation metric since it's essentially a regression problem.

#### Apple

- The RMSE Value after training is 6.103
  ![apple-stock-pred-graph](https://github.com/alvionna/stock-market-prediction/blob/main/images/apple_stock_pred.png)

#### Microsoft

- The RMSE Value after training is 16.946
  ![microsoft-stock-pred-graph](https://github.com/alvionna/stock-market-prediction/blob/main/images/microsoft_stock_pred.png)

#### Google

- The RMSE Value after training is 4.065
  ![google-stock-pred-graph](https://github.com/alvionna/stock-market-prediction/blob/main/images/google_stock_pred.png)

#### Amazon

- The RMSE Value after training is 6.969
  ![amazon-stock-pred-graph](https://github.com/alvionna/stock-market-prediction/blob/main/images/amazon_stock_pred.png)
