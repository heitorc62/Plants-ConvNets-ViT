import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

import webbrowser
import pandas as pd
from os import listdir, getcwd, mkdir
from os.path import join, isdir, exists
from shutil import copy2

def get_projects_list():
    global projects_list

    projects_folder = join(getcwd(), 'main', 'assets')
    projects_list = [f for f in listdir(projects_folder) if isdir(join(projects_folder, f))]
    projects_list.sort()

def update_classes_project(batches_list):
    global classes_per_batch

    temp_classes_project = {}

    for file in batches_list:
        dict_temp = classes_per_batch[file]
        for keys in dict_temp.keys():
            if keys not in temp_classes_project.keys():
                temp_classes_project[keys] = dict_temp[keys]
            else:
                temp_classes_project[keys] += dict_temp[keys]

    temp_keys = list(temp_classes_project.keys())
    temp_keys.sort()
    counts = [temp_classes_project[k] for k in temp_keys]
    classes_text = [{'label': k + '(' + str(c) + ')', 'value': k} for k, c in zip(temp_keys, counts)]
    classes_list = [k for k in temp_keys]
    
    return classes_text, classes_list

def get_classes_per_batch(batches_folder, batches_list):
    global classes_per_batch, loadbar_classes
    
    count = 0
    temp_list = [l for l in batches_list if l not in classes_per_batch.keys()]

    for file in temp_list:
        count += 1
        loadbar_classes = 100*count/len(temp_list)

        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        dict_temp = dict(df_temp['manual_label'].value_counts())
        classes_per_batch[file] = dict_temp 

def save_dataset(output_folder, project_name, selected_classes, selected_batches):
    global loadbar_save_dataset

    count = 0
    project_folder = join(getcwd(), 'main', 'assets', project_name)
    batches_folder = join(project_folder, 'dataframes')
    images_folder = join(project_folder, 'images')
    
    if not exists(output_folder):
        mkdir(output_folder)
    
    for label in selected_classes:
        label_path = join(output_folder, label)
        if not exists(label_path):
            mkdir(label_path)

    for batch in selected_batches:
        count += 1
        loadbar_save_dataset = 100*count/len(selected_batches) 

        batch_folder = batch[:-4]
        match = project_name + '.csv'
        if match in batch: #if project name is included in the csv name
            match_size = len(match) + 1
            batch_folder = batch[:-match_size]

        temp_images_folder = join(images_folder, batch_folder)
        batch_images_folder = join(temp_images_folder, listdir(temp_images_folder)[0])

        csv_path = join(batches_folder, batch)
        df_temp = pd.read_csv(csv_path)
        df_filtered = df_temp.loc[df_temp['manual_label'].isin(selected_classes)]

        for _, row in df_filtered.iterrows():
            name = row['names']
            label = row['manual_label']
            label_path = join(output_folder, label)
            copy2(join(batch_images_folder, name), join(label_path, name))
    loadbar_save_dataset = 0  
    return 0

def get_datasets_list():
    default_output_folder = join(getcwd(), 'output')
    datasets_list = [f for f in listdir(default_output_folder) if isdir(join(default_output_folder, f))]
    return datasets_list

projects_list = []
classes_per_batch = {}
loadbar_classes = 0
loadbar_save_dataset = 0
get_projects_list()
datasets_list = get_datasets_list()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0'
        }
    ]
)

app.layout = html.Div([
    ## header
    html.H1('ILT Data Explorer'),

    dcc.ConfirmDialog(
        id='confirm_save_dataset'
    ),

    html.Datalist(
        id='list_suggested_datasets', 
        children=[html.Option(value=name) for name in datasets_list]
    ),

    ## project selection
    dbc.Container(
        dbc.Row([
            dbc.Col(dcc.Dropdown(projects_list, '', id='dropdown_project', clearable=False), width={"size": 12}),
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Container(
        dbc.Row([
            dbc.Col(
                dcc.Checklist(
                    options = [],
                    value = [],
                    id = 'checklist_batches',
                    labelStyle={'display': 'block'},
                    style={"height": 600, "overflow":"auto"}
                ), width = 6
            ),
            dbc.Col([
                dbc.Button('Display classes', n_clicks=0, id='button_display_classes', style={'background':'chocolate', 'width':'100%'}),
                dbc.Progress(value=0, id='progress_classes'),
                dcc.Checklist(
                    options = [],
                    value = [],
                    id = 'checklist_classes',
                    labelStyle={'display': 'block'},
                    style={"height": 600, "overflow":"auto"}
                )], width = 6
            ),
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Input(
                    value='dataset',
                    id='input_save_dataset',
                    type="text",
                    style={'width': '100%', 'background':'Floralwhite'},
                    list = 'list_suggested_datasets'
                ),
            ),
            dbc.Col([
                dbc.Button('Save Dataset', n_clicks=0, id='button_save_dataset', style={'background':'chocolate', 'width':'100%'}),
                dbc.Progress(value=0, id="progress_save_dataset"),
                dbc.Alert(children = '' , id="alert_save_dataset_problem", dismissable=True, is_open=False, color='warning', duration=5000),
                dbc.Alert(children = '' , id="alert_save_dataset_success", dismissable=True, is_open=False),
            ])
        ])
    ),

    dcc.Interval(id='clock', interval=1000, n_intervals=0, max_intervals=-1),
    dcc.Store(id='button_display_classes_nclicks', data=0),
    dcc.Store(id='dataset_name_ok', data=False),
    dcc.Store(id='save_dataset_result', data=0),
])

"""
    Updates the loadbar when the button_display_classes is pressed
"""
@app.callback(
    Output("progress_classes", "value"),
    Output("progress_save_dataset", "value"),
    Input("clock", "n_intervals"))
def progress_classes_update(n):
    global loadbar_classes, loadbar_save_dataset
    return (loadbar_classes, ), (loadbar_save_dataset, )

"""

"""
@app.callback(
    Output('checklist_classes', 'options'),
    Output('checklist_classes', 'value'),
    Output('button_display_classes_nclicks', 'data'),
    Input('button_display_classes', 'n_clicks'),
    Input('checklist_batches', 'value'),
    State('dropdown_project', 'value'),
    State('button_display_classes_nclicks', 'data')
)
def update_classes_list(nclicks, checklist_value, project_name, prev_nclicks):
    if nclicks > prev_nclicks and len(checklist_value) > 0:
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')

        get_classes_per_batch(batches_folder, checklist_value)

        classes_text, classes_list = update_classes_project(checklist_value)

        return classes_text, classes_list, nclicks
    return [], [], nclicks

@app.callback(
    Output('checklist_batches', 'options'),
    Output('checklist_batches', 'value'),
    Output('input_save_dataset', 'value'),
    Input('dropdown_project', 'value')
)
def update_batches_list(project_name):
    global classes_per_batch, loadbar_classes, loadbar_save_dataset

    if project_name != '':
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
        batches_list = [f for f in listdir(batches_folder) if f[-4:] == '.csv']
        batches_list.sort()

        batches_text = []
        for i in range(len(batches_list)):
            batches_text.append({
                'label': ' ' + str(i+1) + ': ' + batches_list[i],
                'value': batches_list[i]
            })

        classes_per_batch = {}
        loadbar_classes = 0
        loadbar_save_dataset = 0

        return batches_text, batches_list, project_name + '_dataset'
    return [], [], 'dataset'


@app.callback(
    Output('alert_save_dataset_success', 'children'),
    Output('alert_save_dataset_success', 'is_open'),
    Input('confirm_save_dataset', 'submit_n_clicks'),
    State('dropdown_project', 'value'),
    State('input_save_dataset', 'value'),
    State('checklist_batches', 'value'),
    State('checklist_classes', 'value'),
)
def save_dataset_confirmed(nclicks, project_name, dataset_name, selected_batches, selected_classes):
    if nclicks:
        default_output_folder = join(getcwd(), 'output')
        output_folder = join(default_output_folder, dataset_name)

        save_dataset(output_folder, project_name, selected_classes, selected_batches)
        return dataset_name + ' saved successfully', True
    return '', False

@app.callback(
    Output('confirm_save_dataset', 'message'),
    Output('confirm_save_dataset', 'displayed'),
    Input('dataset_name_ok', 'data'),
    State('input_save_dataset', 'value'),
)
def save_dataset_confirmation(name_ok, dataset_name):
    default_output_folder = join(getcwd(), 'output')
    output_folder = join(default_output_folder, dataset_name)

    if name_ok == True:
        if exists(output_folder):
            return dataset_name + ' already exists. Do you really want to merge these images into the existing dataset?', True
        else:
            return dataset_name + ' will be created. Proceed?', True
    return '', False


@app.callback(
    Output('alert_save_dataset_problem', 'children'),
    Output('alert_save_dataset_problem', 'is_open'),
    Output('dataset_name_ok', 'data'),
    Input('button_save_dataset', 'n_clicks'),
    State('dropdown_project', 'value'),
    State('checklist_batches', 'value'),
    State('checklist_classes', 'value'),
)
def click_save_dataset(nclicks, project_name, selected_batches, selected_classes):
    if nclicks > 0:
        default_output_folder = join(getcwd(), 'output')
        if not exists(default_output_folder):
            mkdir(default_output_folder)

        if project_name == '':
            return 'No project was selected', True, False
        elif len(selected_batches) == 0:
            return 'No batches were selected', True, False
        elif len(selected_classes) == 0:
            return 'No classes were selected', True, False
        else:
            return '', False, True
    return '', False, False

port = 8030
opened = False

if not opened:
    webbrowser.open('http://127.0.0.1:' + str(port) + '/', new=2, autoraise=True)
    opened = True

if __name__ == '__main__':
    app.title = 'ILT Data Explorer'
    app.run_server(debug=False, port=port)
