# Stock Analysis 

## Historical Data 
* Using yfinance, gets historical stock data of any inputed ticker
* Looks at most recent 14 day period and finds the most similar 14 day periods in the last twenty years
* Shows headlines of those similar periods to help identify patterns and trends
* Scale past periods to compare to most recent period
* Built using Streamlit for modern UI and clear visualizations of data

## Bollinger Bands

Contains Python code for backtesting the Bollinger Bands Double Bottom trading strategy using the `pandas`, `plotly.graph_objects`, `numpy`, `streamlit`, and `yfinance` libraries. The code aims to identify bullish "bottom w" patterns and generate buy (LONG) or sell (SHORT) signals based on specific conditions.

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

Please ensure that you have the required libraries (`pandas`, `plotly`, `numpy`, `streamlit`, and `yfinance`) installed before running the code.

Example: 
![CleanShot 2023-07-21 at 14 43 12@2x](https://github.com/adigoel4/stockAnalysis/assets/115904374/cc8573de-fd73-4973-bf56-e618227993a5)

