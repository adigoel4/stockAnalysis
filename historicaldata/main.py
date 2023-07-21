import streamlit as st
import yfinance as yf
import pandas as pd

import datetime
import matplotlib.pyplot as plt
import model

from sklearn.metrics.pairwise import cosine_similarity

# CSS
st.markdown(
    """
    <style type="text/css" media="print">
      hr
      {
        page-break-after: always;
        page-break-inside: avoid;
      }
    </style>
""",
    unsafe_allow_html=True,
)


# Functions
def compute_similarity(window):
    last_n_days = df["Close"].iloc[-n_window:]
    similarity_scores = cosine_similarity(
        last_n_days.values.reshape(1, -1), window.values.reshape(1, -1)
    )
    return similarity_scores[0]


def plot_dataframes(df1, df2):
    fig, ax = plt.subplots()
    ax.plot(range(1, 22), df1, label="Last 14 days")
    ax.plot(range(1, 22), df2, label="Similar case from the past")
    ax.set_title("Stock quotes similarity")
    ax.legend()
    st.pyplot(fig)


def create_null_series(length):
    return pd.Series([None] * length)


def adjust_dataframe(df_past, df, adjust_index):
    return df_past * (df["Close"].iloc[adjust_index:][0] / df_past[0])


# title
st.title("Historical Stock Data Analysis")

ticker = st.text_input("Which stock do you want to analyze?").upper()
years = st.slider("How many years of data take into account?", 1, 20, 20)

NUM_DAYS = years * 365

start = datetime.date.today() - datetime.timedelta(NUM_DAYS)
end = datetime.datetime.today()

if len(ticker) > 0:
    n_window = 14
    df = yf.download(ticker, start, end)

    data = model.predictions(df.copy())

    df.reset_index(inplace=True)
    df = df.set_index("Date")
    df = df[-years * 365 :]
    rolling_window = df["Close"].rolling(window=n_window)

    similarity_scores = rolling_window.apply(compute_similarity, raw=False)
    similarity_scores = similarity_scores.fillna(value=0)
    top_similarities = similarity_scores.argsort()[-10:]

    i = 1
    df_past_list = []
    df_past_list_last = []
    for index in top_similarities:
        null_series = create_null_series(7)  # to get 21 days of historical data

        index = index - (n_window - 1)  # starting index is 14 days prior

        df1 = df["Close"].iloc[-n_window:]
        df1_plus_nulls = pd.concat([df1, null_series])
        df_past = df["Close"].iloc[index : index + 21]
        df_past_adj = df_past * (df["Close"].iloc[-14:][0] / df_past[0])
        df_past_adj_last = df_past * (df["Close"].iloc[-1:][0] / df_past[13])

        # append lists
        df_past_list.append(df_past_adj.values.tolist())
        df_past_list_last.append(df_past_adj_last.values.tolist())

        # take into account only periods that large enough to take a peek into the future
        if len(df_past) == 21:
            # ----- front end ----

            st.markdown("---")
            st.write("**Case " + str(i) + "**")
            adjusted = st.checkbox("Scale result?", key=index)

            if adjusted:
                # df_past = df_past * (df["Close"].iloc[-14:][0] / df_past[0])
                df_past = adjust_dataframe(df_past, df, -14)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Last 14 days**")
                st.write(df1)
            with col2:
                st.write("**Similar case from the past**")
                st.write(df_past)
            with col3:
                # Plot the datasets
                st.write("**Plot**")
                plot_dataframes(df1_plus_nulls, df_past)
                st.write("Similarity score")
                st.write(similarity_scores.iloc[index + 13])
            st.write(
                "**Major events that affected the stock market in "
                + str(df.iloc[index].name)[0:7]
                + "**"
            )
            st.write("No data Found.")
            i += 1

    st.markdown("---")
    st.write("**Results summarized on one chart (scaled to the first period)**")
    df_past_list = pd.DataFrame(df_past_list)
    # st.write(df_past_list)
    fig, ax = plt.subplots()
    for i in range(0, len(df_past_list)):
        ax.plot(range(1, 22), df_past_list.iloc[i], label="Last 14 days", alpha=0.2)
    ax.plot(
        range(1, 22),
        df1_plus_nulls,
        label="Last 14 days",
        color="red",
        marker="o",
        markersize=3,
    )
    # Set the chart title and legend
    ax.set_title("Stock quotes similarity")
    ax.grid(color="grey", linestyle="-", linewidth=0.1, axis="y")

    # Display the chart using Streamlit
    st.pyplot(fig)

    if "projection" not in st.session_state:
        st.session_state.projection = [None] * 21

    st.write("**Results summarized on one chart (scaled to the last period)**")

    df_past_list_last = pd.DataFrame(df_past_list_last)
    fig, ax = plt.subplots()
    for i in range(0, len(df_past_list_last)):
        ax.plot(
            range(1, 22), df_past_list_last.iloc[i], label="Last 14 days", alpha=0.2
        )
    ax.plot(
        range(1, 22),
        df1_plus_nulls,
        label="Last 14 days",
        color="red",
        marker="o",
        markersize=3,
    )

    # Set the chart title and legend
    ax.set_title("Stock quotes similarity")
    ax.grid(color="grey", linestyle="-", linewidth=0.1, axis="y")

    # Display the chart using Streamlit
    st.pyplot(fig)
