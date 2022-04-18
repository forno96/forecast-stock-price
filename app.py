import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc

import flask
from os import getenv

try:
    from forecastPrediction import get_df, train_evaluate_data, show_data
except ImportError:
    from .forecastPrediction import get_df, train_evaluate_data, show_data

server = flask.Flask(__name__)
server.secret_key = getenv('secret_key', 'jcsnnfiu3908fu2fowkmsnrwvjok9sp')

app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

stocks = {
    'GOOGL': 'Google', 'AAPL': 'Apple', 'MSFT': 'Microsoft', 'FB': 'Meta', 'TWTR': 'Twitter', 'TSLA': 'Tesla', 'KO': 'Coca Cola'
}
stock_options = [{'label': label, 'value': code} for code, label in stocks.items()]
models = {
    'Lasso': 'Lasso', 'LinearRegression': 'Linear Regression', 'Ridge': 'Ridge'
}
model_options = [{'label': label, 'value': code} for code, label in models.items()]

app.layout = html.Div([
    html.H1('Stock Forecast'),
    html.Div([
        html.Div([
            html.P(["Stock"], className="mb-0"),
            dcc.Dropdown(
                id='STOCK',
                options=stock_options,
                value=stock_options[0]['value'],
                clearable=False
            )], className='col'
        ),
        html.Div([
            html.P(["Model"], className="mb-0"),
            dcc.Dropdown(
                id='MODEL',
                options=model_options,
                value=model_options[0]['value'],
                clearable=False
            )], className='col'
        ),
        html.Div([
            dbc.Button("Recalculate", id="recalculate", color="primary", className="me-1")
            ], className="col-auto"
        ),
    ], className="row align-items-end"),
    dcc.Graph(id='stock-graph', style={'height': '89vh'}, config={'displaylogo': False})
], className="container")

@app.callback(
    Output('stock-graph', 'figure'),
    Input('STOCK', 'value'),
    Input('MODEL', 'value'),
    Input("recalculate", 'n_clicks')
)
def update_graph(STOCK, model_type, n_clicks):
    df = get_df(STOCK)
    test_score, set_score, predict, model = train_evaluate_data(
        df, model_type=model_type
    )
    return show_data(STOCK, model.__class__.__name__, df, test_score, set_score, predict)

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=5000)