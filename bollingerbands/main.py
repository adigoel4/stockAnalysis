import pandas as pd
import plotly.graph_objects as go
import copy
import numpy as np
import streamlit as st
import yfinance as yf

#Calculates bollinger bands 
def bollinger_bands(df):
    data = copy.deepcopy(df)
    data["std"] = data["price"].rolling(window=20, min_periods=20).std()
    data["mid band"] = data["price"].rolling(window=20, min_periods=20).mean()
    data["upper band"] = data["mid band"] + 2 * data["std"]
    data["lower band"] = data["mid band"] - 2 * data["std"]

    return data

# the W is created with 5 nodes (l, k, j, m, i) (order they are found below)
# there are four conditions that indicate a double bottom 

# condition 1: reaction low near Lower Band or breaks it.
# condition 2: price moves around SMA (middle band).
# condition 3: lower low forms without breaking Lower Band
# condition 4: strong move towards SMA, possible breakout

def signal_generation(data, method):
    period = 75

    alpha = 0.03 # difference between the price and the bands (also directly correlated to how often trades trigger)
    beta = 0.03 # beta is used to measure the bandwidth 

    df = method(data) #adding the bands to the df 
    df["signals"] = 0 #when trades are happening

    df["cumsum"] = 0 #holding position
    df["coordinates"] = "" #to draw the double bottom on the graph

    for i in range(period, len(df)):
        moveon = False
        threshold = 0.0 #value of the k node

        #condition 4
        if (df["price"][i] > df["upper band"][i]) and (df["cumsum"][i] == 0): 
            for j in range(i, i - period, -1):
                # condition 2
                if (np.abs(df["mid band"][j] - df["price"][j]) < alpha) and (
                    np.abs(df["mid band"][j] - df["upper band"][i]) < alpha
                ):
                    moveon = True
                    break

            if moveon == True:
                moveon = False
                for k in range(j, i - period, -1):
                    # condition 1
                    if np.abs(df["lower band"][k] - df["price"][k]) < alpha:
                        threshold = df["price"][k]
                        moveon = True
                        break

            if moveon == True:
                moveon = False
                for l in range(k, i - period, -1):
                    # this one is for plotting W shape
                    if df["mid band"][l] < df["price"][l]:
                        moveon = True
                        break

            if moveon == True:
                moveon = False
                for m in range(i, j, -1):
                    # condition 3
                    if (
                        (df["price"][m] - df["lower band"][m] < alpha)
                        and (df["price"][m] > df["lower band"][m])
                        and (df["price"][m] < threshold)
                    ):
                        df.at[i, "signals"] = 1
                        df.at[i, "coordinates"] = "%s,%s,%s,%s,%s" % (l, k, j, m, i)
                        df["cumsum"] = df["signals"].cumsum()
                        moveon = True
                        break

        # sell our position when there is contraction on bollinger bands (indicating momentum ending)
        if (df["cumsum"][i] != 0) and (df["std"][i] < beta) and (moveon == False):
            df.at[i, "signals"] = -1
            df["cumsum"] = df["signals"].cumsum()

    return df


# In[5]:

#visualization
def plot(data):
    trades = list(data[data["signals"] != 0].iloc[:].index)
    # no trades were found
    if len(trades) < 2: 
        st.write("No viable trades found")
    else:
        #create different graphs for each trade
        options = {i for i in range(1,len(trades)//2+1)}
        trade = st.selectbox("Which trade would you like to see?", options)
        if trade is not None:

            a,b = trades[(trade - 1) * 2:trade * 2]
            tplot = data[a - 85 : b + 30] #range for clear graph
            tplot.set_index(
                pd.to_datetime(tplot["date"], format="%Y-%m-%d %H:%M"), inplace=True
            )

            fig = go.Figure()

            #plot stock price
            fig.add_trace(
                go.Scatter(x=tplot.index, y=tplot["price"], mode="lines", name="price")
            )

            #plot upper band
            fig.add_trace(
                go.Scatter(
                    x=tplot.index,
                    y=tplot["upper band"],
                    fill=None,
                    mode="lines",
                    name="upper band",
                )
            )

            #plot lower band
            fig.add_trace(
                go.Scatter(
                    x=tplot.index,
                    y=tplot["lower band"],
                    fill="tonexty",
                    mode="lines",
                    name="lower band",
                )
            )

            #plot middle band
            fig.add_trace(
                go.Scatter(
                    x=tplot.index, y=tplot["mid band"], mode="lines", name="moving average"
                )
            )

            #plot long position
            fig.add_trace(
                go.Scatter(
                    x=tplot[tplot["signals"] == 1].index,
                    y=tplot["price"][tplot["signals"] == 1],
                    mode="markers",
                    marker_symbol="triangle-up",
                    marker=dict(
                        color='green'
                    ),
                    name="LONG",
                )
            )

            #plot short position
            fig.add_trace(
                go.Scatter(
                    x=tplot[tplot["signals"] == -1].index,
                    y=tplot["price"][tplot["signals"] == -1],
                    mode="markers",
                    marker_symbol="triangle-down",
                    marker=dict(
                        color='green'
                    ),
                    name="SHORT",
                )
            )

            #plot W pattern
            temp = tplot["coordinates"][tplot["signals"] == 1]
            indexlist = list(map(int, temp[temp.index[0]].split(",")))

            fig.add_trace(
                go.Scatter(
                    x=tplot["price"][pd.to_datetime(data["date"].iloc[indexlist])].index,
                    y=tplot["price"][pd.to_datetime(data["date"].iloc[indexlist])],
                    mode="lines",
                    name="double bottom pattern",
                )
            )

            fig.update_layout(title=f"Bollinger Bands Trade {trade}", yaxis_title="Price", xaxis_title="Time")

            st.plotly_chart(fig, use_container_width=True)

   
    



#frontend
st.title("Bollinger Bands Double Bottom Backtesting")

#data ingestion
ticker = st.text_input("Which stock do you want to analyze?").upper()
days = st.slider("How many days of data take into account?", 1, 7, 1)

if len(ticker) > 0:
    df = yf.download(ticker, period=f'{days}d', interval='1m')
    df.reset_index(inplace=True)
    data = df[['Datetime', 'Close']]
    data = data.rename(columns={"Datetime": "date", "Close": "price"})
    signals = signal_generation(data, bollinger_bands)
    st.write(signals)
    plot(signals)

                   
