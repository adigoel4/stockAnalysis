# coding: utf-8

# In[1]:

# bollinger bands is a simple indicator
# just moving average plus moving standard deviation
# but pattern recognition is a differenct case
# visualization is easy for human to identify the pattern
# but for the machines, we gotta find a different approach
# when we talk about pattern recognition these days
# people always respond with machine learning
# why machine learning when u can use arithmetic approach
# which is much faster and simpler?

# there are many patterns for recognition
# top m, bottom w, head-shoulder top, head-shoulder bottom, elliott waves
# in this content, we only discuss bottom w
# top m is just the reverse of bottom w
# rules of bollinger bands and bottom w can be found in the following link:
# https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

import pandas as pd
import plotly.graph_objects as go
import copy
import numpy as np
import streamlit as st


# In[2]:

# In[3]:


# first step is to calculate moving average and moving standard deviation
# we plus/minus two standard deviations on moving average
# we get our upper, mid, lower bands
def bollinger_bands(df):
    data = copy.deepcopy(df)
    data["std"] = data["price"].rolling(window=20, min_periods=20).std()
    data["mid band"] = data["price"].rolling(window=20, min_periods=20).mean()
    data["upper band"] = data["mid band"] + 2 * data["std"]
    data["lower band"] = data["mid band"] - 2 * data["std"]

    return data


# In[4]:


# the signal generation is a bit tricky
# there are four conditions to satisfy
# for the shape of w, there are five nodes
# from left to right, top to bottom, l,k,j,m,i
# when we generate signals
# the iteration node is the top right node i, condition 4
# first, we find the middle node j, condition 2
# next, we identify the first bottom node k, condition 1
# after that, we point out the first top node l
# l is not any of those four conditions
# we just use it for pattern visualization
# finally, we locate the second bottom node m, condition 3
# plz refer to the following link for my poor visualization
# https://github.com/je-suis-tm/quant-trading/blob/master/preview/bollinger%20bands%20bottom%20w%20pattern.png
def signal_generation(data, method):
    # according to investopedia
    # for a double bottom pattern
    # we should use 3-month horizon which is 75
    period = 75

    # alpha denotes the difference between price and bollinger bands
    # if alpha is too small, its unlikely to trigger a signal
    # if alpha is too large, its too easy to trigger a signal
    # which gives us a higher probability to lose money
    # beta denotes the scale of bandwidth
    # when bandwidth is larger than beta, it is expansion period
    # when bandwidth is smaller than beta, it is contraction period
    alpha = 0.0001
    beta = 0.0001

    df = method(data)
    df["signals"] = 0

    # as usual, cumsum denotes the holding position
    # coordinates store five nodes of w shape
    # later we would use these coordinates to draw a w shape
    df["cumsum"] = 0
    df["coordinates"] = ""

    for i in range(period, len(df)):
        # moveon is a process control
        # if moveon==true, we move on to verify the next condition
        # if false, we move on to the next iteration
        # threshold denotes the value of node k
        # we would use it for the comparison with node m
        # plz refer to condition 3
        moveon = False
        threshold = 0.0

        # bottom w pattern recognition
        # there is another signal generation method called walking the bands
        # i personally think its too late for following the trend
        # after confirmation of several breakthroughs
        # maybe its good for stop and reverse
        # condition 4
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
                    # this one is for plotting w shape
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

        # clear our positions when there is contraction on bollinger bands
        # contraction on the bandwidth is easy to understand
        # when price momentum exists, the price would move dramatically for either direction
        # which greatly increases the standard deviation
        # when the momentum vanishes, we clear our positions

        # note that we put moveon in the condition
        # just in case our signal generation time is contraction period
        # but we dont wanna clear positions right now
        if (df["cumsum"][i] != 0) and (df["std"][i] < beta) and (moveon == False):
            df.at[i, "signals"] = -1
            df["cumsum"] = df["signals"].cumsum()

    return df


# In[5]:


def plot(new):
    a, b = list(new[new["signals"] != 0].iloc[:2].index)

    newbie = new[a - 85 : b + 30]
    newbie.set_index(
        pd.to_datetime(newbie["date"], format="%Y-%m-%d %H:%M"), inplace=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=newbie.index, y=newbie["price"], mode="lines", name="price")
    )

    fig.add_trace(
        go.Scatter(
            x=newbie.index,
            y=newbie["upper band"],
            fill=None,
            mode="lines",
            name="upper band",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=newbie.index,
            y=newbie["lower band"],
            fill="tonexty",
            mode="lines",
            name="lower band",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=newbie.index, y=newbie["mid band"], mode="lines", name="moving average"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=newbie[newbie["signals"] == 1].index,
            y=newbie["price"][newbie["signals"] == 1],
            mode="markers",
            marker_symbol="triangle-up",
            name="LONG",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=newbie[newbie["signals"] == -1].index,
            y=newbie["price"][newbie["signals"] == -1],
            mode="markers",
            marker_symbol="triangle-down",
            name="SHORT",
        )
    )

    temp = newbie["coordinates"][newbie["signals"] == 1]
    indexlist = list(map(int, temp[temp.index[0]].split(",")))

    fig.add_trace(
        go.Scatter(
            x=newbie["price"][pd.to_datetime(new["date"].iloc[indexlist])].index,
            y=newbie["price"][pd.to_datetime(new["date"].iloc[indexlist])],
            mode="lines",
            name="double bottom pattern",
        )
    )

    fig.update_layout(title="Bollinger Bands Pattern Recognition", yaxis_title="price")

    st.plotly_chart(fig)


# In[6]:

# ta-da
df = pd.read_csv("gbpusd.csv")

signals = signal_generation(df, bollinger_bands)
st.write(signals)

new = copy.deepcopy(signals)
plot(new)
