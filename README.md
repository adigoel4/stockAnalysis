# Stock Analysis 

## Historical Data

This Python code enables the analysis of historical stock data using the `streamlit`, `yfinance`, `pandas`, `datetime`, `matplotlib`, `model`, and `cosine_similarity` libraries. The code allows users to input a stock symbol (ticker) and select the number of years of historical data they want to analyze.

### Features

- **Cosine Similarity Analysis:** The code calculates the cosine similarity score between the last 14 days of the stock's closing prices and 21-day windows of past data. This helps identify similar price patterns in the historical data.

- **Interactive Visualization:** The code provides interactive visualizations using `plotly.graph_objects` and `streamlit`. Users can select specific trades to view the price series, Bollinger Bands, and the identified "bottom w" patterns.

- **Adjustable Scaling:** Users can adjust the scaling of historical data to align it with the current price levels, enabling a more accurate visual comparison.

- **Top Similar Historical Periods:** Users can explore the top 10 most similar historical periods, comparing the price patterns and similarity scores.

### Implementation Details

1. **Computing Similarity Score:**
   The code defines a function `compute_similarity(window)` to calculate the cosine similarity score between the last 14 days of the stock's closing prices and 21-day windows of past data. The similarity score is used to identify similar patterns in the stock's historical data.

2. **Plotting Dataframes:**
   The function `plot_dataframes(df1, df2)` generates a plot comparing the closing prices of the last 14 days with those of a similar case from the past, showcasing the stock's price similarity.

3. **Adjusting Dataframes:**
   The function `adjust_dataframe(df_past, df, adjust_index)` adjusts past dataframes to align them with the current price levels, allowing better comparison between different time periods.

### How to Use

1. Enter the stock symbol (ticker) you want to analyze in the input box.
2. Choose the number of years of historical data you want to use for analysis using the slider.
3. The code will download the historical stock data using `yfinance` and analyze the similarity of price patterns between the last 14 days and 21-day windows from the past.
4. The code will present the top 10 most similar historical periods, allowing users to examine the similarity scores and visualize the price patterns for each case.
5. Users can adjust the scaling of the data for a better visual comparison.
6. The interactive plots will display the closing prices for the last 14 days and the selected historical periods, helping users assess the stock's price similarity.

Example:
![CleanShot 2023-07-22 at 00 39 24@2x](https://github.com/adigoel4/stockAnalysis/assets/115904374/44abf41d-15ca-42ae-83ef-7cb291445573)

## Bollinger Bands

Contains Python code for backtesting the Bollinger Bands Double Bottom trading strategy using the `pandas`, `plotly.graph_objects`, `numpy`, `streamlit`, and `yfinance` libraries. The code aims to identify bullish "bottom w" patterns and generate buy (LONG) or sell (SHORT) signals based on specific conditions.

### Features

- **Bollinger Bands Calculation:** The code calculates Bollinger Bands for a given stock's price series, providing valuable information about price volatility and potential trading opportunities.

- **Double Bottom Pattern Recognition:** The code identifies "bottom w" patterns, a potential bullish reversal pattern, by evaluating specific conditions around the Bollinger Bands and the Simple Moving Average.

- **Buy/Sell Signal Generation:** Based on the identified "bottom w" patterns, the code generates buy (LONG) or sell (SHORT) signals to assist traders in making informed trading decisions.

- **Interactive Visualization:** The code offers interactive plot visualization using `plotly.graph_objects` and `streamlit`, allowing users to explore individual trades and their corresponding patterns and signals.

### Implementation Details

1. **Bollinger Bands Calculation**: The code defines a function `bollinger_bands(df)` that calculates the Bollinger Bands (middle band, upper band, lower band) for a given price series using a 20-period rolling window and standard deviation.

2. **Double Bottom Pattern Recognition**: The function `signal_generation(data, method)` identifies the "bottom w" pattern based on four conditions:
   - Condition 1: A reaction low forms near or breaks the Lower Band.
   - Condition 2: Price moves around the Simple Moving Average (SMA), which is the Middle Band.
   - Condition 3: A lower low forms without breaking the Lower Band.
   - Condition 4: A strong move towards the SMA, indicating a possible breakout.

3. **Visualization**: The code provides interactive visualizations using `plotly.graph_objects` and `streamlit`. Users can select specific trades to view the price series, Bollinger Bands, and the identified "bottom w" patterns.

### How to Use

1. Enter the stock symbol (ticker) you want to analyze in the input box.
2. Choose the number of days of historical data you want to use for backtesting with the slider.
3. The code will download the data using `yfinance`, calculate the Bollinger Bands, and generate buy/sell signals.
4. Select which trade you would like to visualize
5. The interactive plot shows the stock price, Bollinger Bands, and trade entry/exit points based on the identified "bottom w" patterns.

Example: 
![CleanShot 2023-07-21 at 14 43 12@2x](https://github.com/adigoel4/stockAnalysis/assets/115904374/cc8573de-fd73-4973-bf56-e618227993a5)

## Disclaimer

This code is intended for educational and informational purposes only and does not constitute financial advice. Users should exercise caution when making investment decisions based on the results. Always consult with a qualified financial advisor before making any investment. The code's ability to identify similarities in historical data does not guarantee future stock performance. Use the code at your own risk.


