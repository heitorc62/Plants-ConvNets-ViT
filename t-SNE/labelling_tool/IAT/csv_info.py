from datetime import datetime, timezone

import dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc

from os import listdir, stat, system
from os.path import join, isdir, isfile
import webbrowser
import pandas as pd

import os
import sys

from threading import Thread

port = 8020

def create_table(path_to_project):
    path = join(path_to_project, 'dataframes')
    file_list = [f for f in listdir(path) if (isfile(join(path, f)) and f.startswith('batch'))]
    file_list.sort()

    modified_list = [datetime.fromtimestamp(stat(join(path, f)).st_mtime).strftime('%d-%m-%Y-%H:%M') for f in file_list]
    df = pd.DataFrame(list(zip(file_list, modified_list)), columns =['File', 'Last modified'])

    return df

assets_path = 'main/assets/'
df = create_table(join(assets_path, 'lroot'))

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True
                )

header = html.H1("Image Labeling Tool", style={'color': 'CornflowerBlue'})
project_list = [f for f in listdir(assets_path) if isdir(join(assets_path, f))]


app.layout = html.Div([
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        header
                    ),
                ),
            ],
            style={'max-height': '128px','color': 'white',}
        ),
        className='d-flex align-content-end',
        style={'max-width': '100%','background-color': 'Floralwhite'},
    ),
    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Container(
        dbc.Row([
            html.Div('Project: '),
            dcc.Dropdown(project_list, project_list[0], id='project_dropdown', clearable=False),
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Row(
        dbc.Col(
            html.Div(
                id = 'table_div',
                className = 'table_div',
                style=dict(height='62vh', overflow='scroll'),
            ),
        width={"size": 8, "offset": 2}),
    ),
    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Row(
        dbc.Col(
            dbc.Button('Select a project and batch to start labeling', id='start_button', n_clicks=0, disabled=True, style={'width': '100%'}),
        width={"size": 8, "offset": 2}),
    ),
    dcc.Store(id='selected_batch', data=None),
    dcc.Store(id='dummy', data=None),
    dcc.Store(id='df_store', data=None),

])

@app.callback(
    Output('df_store', 'data'),
    Output('table_div', 'children'),
    Input('project_dropdown', 'value')
)
def update_table(value):
    df = create_table(join(assets_path, value))

    return df.to_json(), html.Div([
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], id='table')
    ])

@app.callback(
    Output('selected_batch', 'data'),
    Output('start_button', 'disabled'),
    Output('start_button', 'children'),
    State('df_store', 'data'),
    Input('table', 'active_cell'))
def update_selected(json_df, active_cell):
    df = pd.read_json(json_df)
    if isinstance(active_cell, dict):
        batch = df['File'].tolist()[active_cell['row']]

        return (batch[:-4], False, 'Click here to start labeling ' + batch)
    else:
        return ('', True, 'Select a project and batch to start labeling')

@app.callback(
    Output('dummy', 'data'),
    State('project_dropdown', 'value'),
    State('selected_batch', 'data'),
    Input('start_button', 'n_clicks'))
def start_labeling(selected_project, selected_batch, n):
    if isinstance(selected_project, str) and isinstance(selected_batch, str):
        print(selected_project, selected_batch)
        path_to_images = join('main/assets/', selected_project, 'images', selected_batch)
        path_to_csv = join('main/assets/', selected_project, 'dataframes', selected_batch + '.csv')

        system('python main/app.py ' + path_to_images + ' ' + path_to_csv)
    return 0

app.title = 'IAT Menu'

opened = False
if not opened:
    webbrowser.open('http://127.0.0.1:' + str(port) + '/', new=2, autoraise=True)
    opened = True

if __name__ == '__main__':
    app.run_server(debug=False, port=port)

