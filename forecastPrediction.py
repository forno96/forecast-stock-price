import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from urllib.parse import urlencode
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import holidays

HOLIDAYS_US = holidays.US()


def next_business_day(day):
    next_day = day + timedelta(days=1)
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += timedelta(days=1)

    return next_day


def get_df(STOCK, base_url='https://query1.finance.yahoo.com/v7/finance/download/', start_date=None, end_date=None):
    if ".csv" in STOCK:
        query_url = STOCK
    else:
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date.replace(year=end_date.year-1)

        url = f"{base_url}{STOCK}?"
        params = {
            'period1': int(start_date.timestamp()),
            'period2': int(end_date.timestamp()),
            'interval': "1d",
            'events': 'history',
            'includeAdjustedClose': True
        }
        
        query_url = url + urlencode(params)

    df = pd.read_csv(
        query_url,
        parse_dates=["Date"],
        date_parser=lambda x: datetime.strptime(str(x), "%Y-%m-%d")
    )

    return df


def prepare_data(df, forecast_col=[], forecast_out=5, test_size=0.3):
    # creating new column called label with the last 5 rows are nan
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[forecast_col])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    # creating the column i want to use later in the predicting method
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y,
        test_size=test_size
    )

    return X_train, X_test, Y_train, Y_test, X_lately


def train_evaluate_data(df, forecast_col=['Open', 'Close', 'High', 'Low'], model_type="Lasso"):
    X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col)

    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "Ridge":
        model = Ridge()
    elif model_type == "Lasso":
        model = Lasso()

    model.fit(X_train, Y_train)

    # Evaluate set & test score
    test_score = model.score(X_test, Y_test)
    set_score = model.score(X_train, Y_train)

    # Create predict
    predict = model.predict(X_lately)
    predict = pd.DataFrame(predict, columns=forecast_col)
    # Add date for the predict
    Date = []
    last_day = df['Date'].max()
    for row in predict.iterrows():
        last_day = next_business_day(last_day)
        Date.append(last_day)    
    predict = predict.assign(Date=Date)

    return test_score, set_score, predict, model


def show_data(STOCK, model_name, df, test_score, set_score, predict, sas=True):
    most_recent_data = predict["Date"].max()
    
    fig = go.Figure(
        data=[
            go.Candlestick(
                name="Storic",
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ),
            go.Candlestick(
                name="Predict",
                x=predict['Date'],
                open=predict['Open'],
                high=predict['High'],
                low=predict['Low'],
                close=predict['Close'],
                increasing_line_color='lightgreen', decreasing_line_color='lightsalmon'
            )
        ]
    )
    fig = fig.update_xaxes(range=[
        most_recent_data - timedelta(days=40),
        most_recent_data + timedelta(days=1)
    ])
    fig = fig.update_layout(
        template="plotly_white",
        title=f"""
            <b>{model_name}</b> on <b>{STOCK}</b>
            <br>Training set score: <b>{int(test_score*100)}%</b>
            <br>Test set score: <b>{int(set_score * 100)}%</b>
        """
    )

    if sas is False:
        fig.show()
        print(df)
        print(test_score)
        print(predict)
    
    return fig


if __name__ == '__main__':
    STOCK = "GOOGL"
    df = get_df(STOCK)
    test_score, set_score, predict, model = train_evaluate_data(df)
    show_data(STOCK, df, test_score, set_score, predict, sas=False)
