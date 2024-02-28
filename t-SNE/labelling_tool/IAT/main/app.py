
import sys

assert len(sys.argv) >= 3

from functions import *

import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go

import pandas as pd
pd.options.mode.chained_assignment = None # default='warn'
import pandas_datareader.data as web

import imageselector
import json
import ast

from os import mkdir, getcwd, listdir
from os.path import exists, join, dirname, basename
import shutil
import webbrowser
from timeit import default_timer as timer
from datetime import date

def init(argv):
    """
        Read the arguments from the terminal

        argv (list):
            [1] (string): path to the images
            [2] (string): path to the thumbnails
            [3] (string): input CSV
            [4] (string): n = range of the graph (x = [-n, n], y = [-n, n])
    """
    
    path_to_images = argv[1]
    path_to_thumbnails = argv[2]
    csv_file = argv[3]
    graph_range = 80
    if len(argv) > 4:
        graph_range = int(argv[4])

    print('imgs:', path_to_images)
    print('thumbnails:', path_to_thumbnails)
    print('csv:', csv_file)

    project_name = path_to_images.split('/')[1]
    
    csv_folder = dirname(csv_file)
    csv_basename = basename(csv_file)
    batch_str = csv_basename[:-4]
    background_base = join('main', 'assets', project_name, 'backgrounds')
    background_str = [folder for folder in listdir(background_base) if folder[:-4] == batch_str][0]
    
    background_path = join(background_base, background_str)
    
    user_id = 0
    port = 8025

    return path_to_images, path_to_thumbnails, csv_file, csv_folder, csv_basename, background_path, user_id, port, graph_range

def read_input_csv(csv_file):
    """
        Read the input CSV and returns the respective DataFrame

        csv_file (string): path to the input CSV
    """
    return pd.read_csv(csv_file, encoding='ISO-8859-1')

def init_appearance():
    """
        Sets parameters of the interface
    """

    background_color = 'rgba(255, 250, 240, 100)'
    return background_color

path_to_images, path_to_thumbnails, csv_file, csv_folder, csv_basename, background_path, user_id, port, graph_range = init(sys.argv)
df = read_input_csv(csv_file)
thumbs_jpg = False
if 'thumbnails' in df.columns:
    thumbs_jpg = True 

widths, heights = compute_img_ratios(path_to_images, df['names'])
df['widths'] = widths
df['heights'] = heights

background_ranges = {'x': [-graph_range, graph_range], 'y': [-graph_range, graph_range]}

init_par_coords = False #used for recomputing the intervals of the parcoords

THUMBNAIL_WIDTH = 28
THUMBNAIL_HEIGHT = 28
background_color = init_appearance()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
labels_buffer = None
colors_buffer = None

labels_list = df['manual_label'].tolist()
colors_list = df['colors'].tolist()

labels_colors = {}
for i in range(len(labels_list)):
    labels_colors[labels_list[i]] = colors_list[i]
next_label_id = df['colors'].max()+1

#############################################################################

header = html.H1('Image Labeling Tool - ' + csv_basename[:-4], style={'color': 'CornflowerBlue'})

#button_group_1 = dbc.ButtonGroup(
#    [
#        dbc.Button("Keep", n_clicks=0, id='button_galeria_filtrar', style={'background':'darkgreen'}),
#        dbc.Button("Discard", n_clicks=0, id='button_galeria_excluir', style={'background':'darkred'}),
#        #dbc.Button("Select all", n_clicks=0, id='button_galeria_selecionar_todos', style={'background':'darkslateblue'}),
#    ],
#    vertical=True,
#    className='my-btn-group',
#    size="lg",
#)

button_group_2 = dbc.ButtonGroup(
    [
        dbc.Input(placeholder="enter label here", id='input_aplicar_novo_label', type="text", style={'width': '50%', 'background':'Floralwhite'}),
        dbc.Button("Label",n_clicks=0, id='button_aplicar_novo_label', style={'background':'chocolate', 'width':'50%'}),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_3 = dbc.ButtonGroup(
    [
        dbc.Input(value=csv_basename, id='input_save_csv', type="text", style={'width': '50%', 'background':'Floralwhite'}),
        dbc.Button("Save CSV",n_clicks=0, id='button_save_csv', style={'background':'chocolate', 'width':'50%'}),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_4 = dbc.ButtonGroup(
    [
        dbc.Input(value="annotated_dataset", id='input_finish_work', type="text", style={'width': '50%', 'background':'Floralwhite'}),
        dbc.Button("Save Annotated Dataset", n_clicks=0, id='button_finish_work', style={'background':'chocolate', 'width':'50%'}),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_5 = dbc.ButtonGroup(
    [
        dbc.Button("UNDO", n_clicks=0, id='button_undo', style={'background':'firebrick', 'width':'100%'}),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_6 = dbc.ButtonGroup(
    [
        dbc.Button("Invert marks", n_clicks=0, id='button_invert_marks', style={'background':'chocolate', 'width':'100%'}),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_7 = dbc.ButtonGroup(
    [
        dcc.Checklist([' Save CSV after labeling'], value = [' Save CSV after labeling'], id = 'check_save_csv'),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_8 = dbc.ButtonGroup(
    [
        dcc.Checklist([' Discard images after labeling'], value = [' Discard images after labeling'], id = 'check_discard_marked'),
    ],
    #vertical=True,
    className='my-btn-group',
    size="lg",
)

button_group_9 = dbc.ButtonGroup(
    [
        dcc.Checklist([' Marked images first'], value = [], id = 'check_marked_first'),
        dcc.Checklist([' Hide relabeled images'], value = [], id = 'check_discard_relabeled'),
    ],
    vertical=True,
    className='my-btn-group',
    size="lg",
)

IMAGES = create_list_dics(
    _list_src=[],
    _list_thumbnail=[],
    _list_name_figure=[]
)

#IMAGES = create_list_dics(
#    _list_src=list(path_to_images + df['names']),
#    _list_thumbnail=list(path_to_images + df['names']),
#    _list_name_figure=list(df['names']),
#    _list_thumbnailWidth=[THUMBNAIL_WIDTH] * df.shape[0],
#    _list_thumbnailHeight=[THUMBNAIL_HEIGHT] * df.shape[0],
#    _list_isSelected= [True] * df.shape[0],
#    _list_custom_data=list(df['custom_data']),
#    _list_thumbnailCaption='',
#    _list_tags='')

#fig = f_figure_scatter_plot(df, _columns=['x', 'y'], _selected_custom_data=list(df['custom_data']))
fig = f_figure_scatter_plot(df, _columns=['x', 'y'], _selected_custom_data=[], background_img=background_path,
                            xrange = background_ranges['x'], yrange=background_ranges['y'])

#_columns_paralelas_coordenadas = ['Layer_A', 'Layer_B', 'Layer_C', 'Layer_D', 'Layer_E', 'Layer_F', 'Layer_G']
if df.shape[1] < 30: #v0.2
    if 'D2' in df:
        _columns_paralelas_coordenadas = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    else:
        _columns_paralelas_coordenadas = ['D1', 'x2', 'y2', 'D4', 'x3', 'y3', 'D7']
else: #v0.3
    _columns_paralelas_coordenadas = ['D1', 'D4', 'D7', 'Area (pxl)', 'circularity', 'Elongation', 'Mean Intensity']


fig_paral =  f_figure_paralelas_coordenadas(
                        _df=df,
                        _filtered_df=pd.DataFrame(columns=df.columns),
                        _columns=_columns_paralelas_coordenadas,
                        #_selected_custom_data=list(df['custom_data']),
                        _selected_custom_data=[],
                        _fig = None
                    )

##############################################################################################################

dropdown_image_vals = ['A-Z, a-z', 'Similarity']
if df.shape[1] > 30: #> v0.3 
    dropdown_image_vals.extend( [
        'Image State (T/F)',
        'SegmentationMethod',
        'Area (pxl)',
        'Image Width (pxl)',
        'Image Size (pxl)',
        'circularity',
        'Elongation',
        'Rectangularity',
        'Mean Intensity',
        'Median Intensity',
        'Contrast',
        'Solidity'        
    ])
dropdown_image_vals.sort()

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
        dbc.Row(
            [
                dbc.Col([
                    dbc.Row([
#                        dbc.Col(
#                            dbc.Button("Map of images",n_clicks=0, id='button_map_images', style={'background':'chocolate', 'width':'100%'})
#                        , width={'size': 4}),
                        dbc.Col(
                            html.Div(
                                html.P('Background opacity'),    
                            style={'textAlign': 'left'})
                        , width={'size': 2}),
                        dbc.Col([
                            dcc.Slider(0, 1, 0.1,
                                    value=0,
                                    id='slider_map_opacity',
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ], width={'size': 3}),
#                        dbc.Col(
#                            html.Div()
#                        , width={'size': 1}),

                       dbc.Col(
                            html.Div(
                                html.P('Marker size'),    
                            style={'textAlign': 'right'})
                        , width={'size': 2}),
                        dbc.Col([
                            dcc.Slider(1, 25, 3,
                                    value=10,
                                    id='slider_marker_size',
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ], width={'size': 3}),
#                        dbc.Col(
#                            html.Div()
#                        , width={'size': 1}),
                        dbc.Col(
                            dcc.Dropdown(['A-Z, a-z', 'Frequency'], value='A-Z, a-z', id='dropdown_order_labels', clearable = False)
                        , width={'size': 2}),
                    ], align='bottom'),
                    dcc.Graph(id="g_scatter_plot", figure=fig, style={"height": "74vh"}, config={'displaylogo':False, 'modeBarButtonsToRemove': ['toImage', 'resetScale2d']}),
                    
                ], width={"size": 7}),
                
                dbc.Col([
                    dbc.Row([
                        dbc.Col(button_group_7, width={"size": 6}),
                        dbc.Col(button_group_8, width={"size": 6}),
                    ]),
                    dbc.Row([
                        #dbc.Col(button_group_1, width={"size": 6}),
                        #dbc.Col(button_group_2, width={"size": 6})
                        dbc.Col(button_group_2, width={"size": 12})
                    ]),
                    dbc.Row([dbc.Col(html.Hr()),],),
                    dbc.Row([
                        dbc.Col(button_group_6, width={"size": 4}),
                        dbc.Col(button_group_9, width={"size": 5}),
                        dbc.Col(
                            dcc.Dropdown(dropdown_image_vals, value='Similarity', id='dropdown_order_images', clearable = False)
                        , width={'offset': 0, 'size': 3})
                    ]),

                    dbc.Row(
                        html.Div(imageselector.ImageSelector(id='g_image_selector', images=IMAGES,
                            galleryHeaderStyle = {'position': 'sticky', 'top': 0, 'height': '0px', 'background': "#000000", 'zIndex': -1},),
                            id='XXXXXXXXXX', style=dict(height='63vh',overflow='scroll', backgroundColor=background_color)
                            )
                        ),

                    ], width={"size": 5}),
                    
            ]),
        style={'max-width': '100%'},
        # className='mt-2'
    ),

    dbc.Row([dbc.Col(html.Hr()),],),
    
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="g_coordenadas_paralelas", figure=fig_paral, style={"height": "70vh"}, config={'displayModeBar': False}), style=dict(width='100%',overflow='scroll'), width={"size": 12}),
            ]
        ),
        style={'max-width': '100%'},
    ),

    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Container(
        dbc.Row([
            dbc.Col(button_group_3, width={"size": 4}),
            dbc.Col(width={"size": 1}),
            dbc.Col(button_group_4, width={"size": 4}),
            dbc.Col(width={"size": 1}),
            dbc.Col(button_group_5, width={"size": 2}),
        ]),
        style={'max-width': '100%'},
    ),

    #dcc.Store(id='selected_custom_points', data=json.dumps(list(df['custom_data']))),
    dcc.Store(id='selected_custom_points', data=json.dumps([])),
    dcc.Store(id='unchecked_points', data=json.dumps([])),
    dcc.Store(id='chart_flag', data=0),
    dcc.Store(id='state_store_df', data=df.to_json()),
    dcc.Store(id='dummy_csv_save', data=0),
    dcc.Store(id='dummy_finish_work', data=0),
    dcc.Store(id='dummy_map_images', data=0),
    html.Div(id='output')
]) #End html.DIV

############3

def save_csv(df_updated, csv_name):
    global csv_folder
    filename = join(csv_folder, csv_name)
    df_updated.to_csv(filename, index=False)
    
    backup_path = join(csv_folder, 'backups')
    if not exists(backup_path):
        mkdir(backup_path)
    backup_path = join(backup_path, csv_name[:-4] + '_bkp' + str(date.today()) + '.csv')
    df_updated.to_csv(backup_path, index=False)
    print('\ncsv recorded.\n')

##############################################################################################################

@app.callback(
    [
        Output('dummy_finish_work', 'data'),
    ],
    [
        Input('button_finish_work', 'n_clicks'),
    ],
    [
        State('state_store_df', 'data'),
        State("input_finish_work", "value"),
    ]
    )
def save_dataset(
    i_button_save_csv_nclicks,
    s_store_df,
    s_input_finish_work_value
    ):

    if i_button_save_csv_nclicks > 0:
        df_updated = pd.read_json(s_store_df)

        if not exists (s_input_finish_work_value):
            mkdir(s_input_finish_work_value)

        labels = df_updated['manual_label']
        for label in labels:
            if not exists(join(s_input_finish_work_value, label)):
                mkdir(join(s_input_finish_work_value, label))

        for index, row in df_updated.iterrows():
            name = row['names']
            folder = row['manual_label']
            shutil.copy2('main/' + path_to_images + name, join(s_input_finish_work_value, folder))

        print('\nDataset recorded.\n')
    
    return [0]

@app.callback(
    [
        Output('dummy_csv_save', 'data'),
    ],
    [
        Input('button_save_csv', 'n_clicks'),
    ],
    [
        State('state_store_df', 'data'),
        State("input_save_csv", "value"),
    ]
    )
def button_save_csv(
    i_button_save_csv_nclicks,
    s_store_df,
    s_input_save_csv_value
    ):

    if i_button_save_csv_nclicks > 0:
        df_updated = pd.read_json(s_store_df)

        save_csv(df_updated, str(s_input_save_csv_value))
    
    return [0]

# @app.callback(
#     [
#         Output('dummy_map_images', 'data'),
#     ],
#     [
#         Input('button_map_images', 'n_clicks'),
#     ],
#     [
#         State('state_store_df', 'data'),
#         State('g_scatter_plot', 'figure'),
#     ]
#     )
# def create_map_of_images(
#     i_button_map_images,
#     s_store_df,
#     s_g_scatter_plot_figure
#     ):

#     global path_to_images

#     if i_button_map_images > 0:
#         df_updated = pd.read_json(s_store_df)
#         fig_scatter = go.Figure(s_g_scatter_plot_figure)

#         image_path = map_of_images(df_updated, fig_scatter, path_to_images)
#         webbrowser.open(join(getcwd(), image_path), new=2, autoraise=True)
    
#     return [0]

@app.callback(
    [
        Output('selected_custom_points', 'data'),
        Output('state_store_df', 'data'),
        Output('chart_flag', 'data'),
        Output('unchecked_points', 'data'),
    ],
    [
        Input('g_scatter_plot', 'selectedData'),
        Input('g_coordenadas_paralelas', 'restyleData'),
        #Input('button_galeria_filtrar', 'n_clicks'),
        #Input('button_galeria_excluir', 'n_clicks'),
        Input('button_aplicar_novo_label', 'n_clicks'),
        Input('button_invert_marks', 'n_clicks'),
        Input('button_undo', 'n_clicks'),
    ],
    [
        State('selected_custom_points', 'data'),
        State('unchecked_points', 'data'),
        State('state_store_df', 'data'),
        State('g_coordenadas_paralelas', 'figure'),
        State('g_image_selector', 'images'),
        State('input_aplicar_novo_label', 'value'),
        State('check_save_csv', 'value'),
        State('input_save_csv', 'value'),
        State('check_discard_marked', 'value')
    ]
    )
def mudanca_custom_data(
    i_selection_g_scatter_plot,
    i_g_coordenadas_paralelas_restyleData,
    #i_button_galeria_filtrar_nclicks,
    #i_button_galeria_excluir_nclicks,
    i_button_aplicar_novo_label_nclicks,
    i_invert_marks_nclicks,
    i_button_undo_nclicks,
    s_selected_custom_points,
    s_unchecked_points,
    s_store_df,
    s_g_coordenadas_paralelas_figure,
    s_button_galeria_filtrar_images,
    s_input_aplicar_novo_label_value,
    s_check_save_csv_value,
    s_input_save_csv_value,
    s_check_discard_marked_value
    ):

    #print('app.py entrou no mudanca_custom_data')
    
    global labels_list
    global labels_colors
    global next_label_id
    global labels_buffer # for the undo
    global colors_buffer # for the undo
    global init_par_coords

    init_par_coords = False
    set_chart_flag = 0
    df_updated = pd.read_json(s_store_df)
    ctx = dash.callback_context
    flag_callback = ctx.triggered[0]['prop_id'].split('.')[0]

    df_store_updated = df_updated.to_json()
    selectedpoints = s_selected_custom_points
    unchecked_points = []

    if flag_callback == 'g_scatter_plot':
        #print('app.py entrou na callback do g_scatter_plot')
        set_chart_flag = 0
        if i_selection_g_scatter_plot and i_selection_g_scatter_plot['points']:
            selectedpoints = [d['customdata'] for d in i_selection_g_scatter_plot['points']]
            selectedpoints = json.dumps(selectedpoints)
        else:
            #selectedpoints = json.dumps(list(df_updated['custom_data']))
            selectedpoints = json.dumps([])
        init_par_coords = True

    elif flag_callback == 'g_coordenadas_paralelas':
        #print('app.py entrou na callback g_coordenadas_paralelas')
        set_chart_flag = 1
        selectedpoints = [d['customdata'] for d in i_selection_g_scatter_plot['points']]

        filtered_df = df_updated[df_updated['custom_data'].isin(selectedpoints)]
        
        if i_g_coordenadas_paralelas_restyleData is not None:
            selected_and_checked = update_df_paralelas_coord(
                _df =filtered_df,
                _list_columns =_columns_paralelas_coordenadas,
                _figure = s_g_coordenadas_paralelas_figure,
                _new_dim_vals=i_g_coordenadas_paralelas_restyleData
            )

            for point in s_button_galeria_filtrar_images:
                if point['custom_data'] not in selected_and_checked:
                    unchecked_points.append(point['custom_data'])
            if i_selection_g_scatter_plot:
                selectedpoints = [d['customdata'] for d in i_selection_g_scatter_plot['points']]
            else:
                selectedpoints = []
            selectedpoints = json.dumps(selectedpoints)
        else:
            selectedpoints = json.dumps(list(df_updated['custom_data']))
  
    elif flag_callback == 'button_aplicar_novo_label':
        s_selected_custom_points = ast.literal_eval(s_selected_custom_points)
        selected_and_checked = []
        selected_not_checked = []
        for point in s_button_galeria_filtrar_images:
            if point['isSelected'] == True:
                selected_and_checked.append(point['custom_data'])
            else:
                selected_not_checked.append(point['custom_data'])
        new_label = str(s_input_aplicar_novo_label_value)
        labels_buffer = df_updated['manual_label'].copy() # for the undo
        colors_buffer = df_updated['colors'].copy()  # for the undo
        df_updated['manual_label'][df_updated['custom_data'].isin(selected_and_checked)]= new_label
        if new_label not in labels_list:
            labels_list.append(new_label)
            labels_colors[new_label] = next_label_id
            next_label_id += 1
        df_updated['colors'][df_updated['custom_data'].isin(selected_and_checked)] = labels_colors[new_label]
        
        if s_check_save_csv_value == [' Save CSV after labeling']:
            save_csv(df_updated, str(s_input_save_csv_value))
        
        df_store_updated = df_updated.to_json()
        
        if s_check_discard_marked_value == [' Discard images after labeling']:
            selectedpoints = json.dumps(selected_not_checked)
            
    elif flag_callback == 'button_undo':
        df_updated['manual_label'] = labels_buffer
        df_updated['colors'] = colors_buffer
        df_store_updated = df_updated.to_json()
        selectedpoints = json.dumps([])

    elif flag_callback == 'button_invert_marks':
        selected_and_checked = []
        selected_not_checked = []

        for point in s_button_galeria_filtrar_images:
            if point['isSelected'] == True:
                unchecked_points.append(point['custom_data'])
        df_store_updated = df_updated.to_json()
    else:
        set_chart_flag = 0
        #selectedpoints = json.dumps(list(df_updated['custom_data']))
        selectedpoints = json.dumps([])

    s_unchecked_points = json.dumps(unchecked_points)
    return [selectedpoints, df_store_updated, set_chart_flag, s_unchecked_points]


@app.callback(
    [
    Output('g_scatter_plot', 'figure'),
    Output('g_image_selector', 'images'),
    Output('g_coordenadas_paralelas', 'figure'),
    ],
    [
    Input('selected_custom_points', 'data'),
    Input('unchecked_points', 'data'),
    Input('dropdown_order_labels', 'value'),
    Input('slider_map_opacity', 'value'),
    Input('slider_marker_size', 'value'),
    Input('dropdown_order_images', 'value'),
    Input('check_marked_first', 'value'),
    Input('check_discard_relabeled', 'value'),
    ],
    [
    State('g_scatter_plot', 'figure'),
    State('g_image_selector', "images"),
    State('g_coordenadas_paralelas', 'figure'),
    State('state_store_df', 'data'),
    State('chart_flag', 'data'),
    ]
    )
def scatter_plot_image_selector(
    i_selected_custom_points,
    i_unchecked_points,
    i_dropdown_order_labels_value, 
    i_slider_map_opacity_value, 
    i_slider_marker_size_value, 
    i_dropdown_order_images_value,
    i_check_marked_first_value,
    i_check_discard_relabeled_value,
    s_g_scatter_plot_figure,
    s_g_image_selector_images,
    s_g_coordenadas_paralelas_figure,
    s_store_df,
    s_chart_flag_data,
    ):

    global path_to_images, path_to_thumbnails, background_path, widths, heights, thumbs_jpg

    prev_fig = go.Figure(s_g_scatter_plot_figure)

    ctx = dash.callback_context
    flag_callback = ctx.triggered[0]['prop_id'].split('.')[0]

    _df = pd.read_json(s_store_df)
    print(flag_callback)

    if flag_callback in ['selected_custom_points', 'dropdown_order_labels', 'slider_map_opacity', 'slider_marker_size',
                         'dropdown_order_images', 'check_marked_first', 'check_discard_relabeled']:
   
        opacity_changed = False
        if flag_callback == 'slider_map_opacity':
            opacity_changed = True
        unchecked_points = json.loads(i_unchecked_points)

        #print('app.py entrou na callback selected_custom_points')
        selected_points = json.loads(i_selected_custom_points)
        if init_par_coords:
            init_for_update_pc(selected_points)

        fig_scatter = f_figure_scatter_plot(_df, _columns=['x', 'y'], _selected_custom_data=selected_points, prev_fig = prev_fig, order_by=i_dropdown_order_labels_value,
                                            background_img=background_path, opacity_changed = opacity_changed, opacity = i_slider_map_opacity_value,
                                            marker_size = i_slider_marker_size_value,xrange = background_ranges['x'], yrange=background_ranges['y'])
        
        filtered_df = _df.loc[_df['custom_data'].isin(selected_points)]

        if i_check_discard_relabeled_value == [' Hide relabeled images']:
            filtered_df = filtered_df[filtered_df['manual_label'] == filtered_df['correct_label']]    
        
        if i_dropdown_order_images_value == 'Similarity': # show similar images close to each other
            ordered_df = filtered_df.sort_values(by='D7') 
        elif i_dropdown_order_images_value != 'A-Z, a-z':
            ordered_df = filtered_df.sort_values(by=i_dropdown_order_images_value) 
        else:
            ordered_df = filtered_df.sort_values(by='manual_label') 
            
        if flag_callback in ['dropdown_order_images', 'check_marked_first']:   
            data_temp = [l['custom_data'] for l in s_g_image_selector_images]
            selection_temp = [l['isSelected'] for l in s_g_image_selector_images]
                        
        else:
            data_temp = selected_points
            selection_temp = [(x not in unchecked_points) for x in data_temp]
        
        df_selection = pd.DataFrame(list(zip(data_temp, selection_temp)), columns =['custom_data', 'selected'])               
        
        ordered_df = pd.merge(ordered_df, df_selection, how="inner", on='custom_data')     
        
        if i_check_marked_first_value == [' Marked images first']:
            checked_df = ordered_df[ordered_df['selected'] == True]
            unchecked_df = ordered_df[ordered_df['selected'] == False]
            ordered_df = pd.concat([checked_df, unchecked_df])
            
        _image_teste_list_correct_label = ordered_df['correct_label']
        _image_teste_list_names = ordered_df['names']
        
        if thumbs_jpg:
            _image_teste_list_thumbs = ordered_df['thumbnails']
        else:
            _image_teste_list_thumbs = ordered_df['names']

        
        _image_teste_list_widths = ordered_df['widths']
        _image_teste_list_heights = ordered_df['heights']

        _image_teste_list_caption = ordered_df['manual_label']
        _image_teste_list_custom_data = ordered_df['custom_data']
        _image_teste_list_texts = ['id: ' + str(id) + ' (' + label + ') - ' + name \
            for id, label, name in zip(_image_teste_list_custom_data, _image_teste_list_caption, _image_teste_list_names)]

        _image_teste_list_selection = list(ordered_df['selected'])
        

        #old
        #_image_teste_list_correct_label = updated_df['correct_label'][updated_df['custom_data'].isin(selected_points)]
        #_image_teste_list_names = updated_df['names'][updated_df['custom_data'].isin(selected_points)]
        #_image_teste_list_caption = updated_df['manual_label'][updated_df['custom_data'].isin(selected_points)]
        #_image_teste_list_custom_data = updated_df['custom_data'][updated_df['custom_data'].isin(selected_points)]
        
        fig2 = create_list_dics(
            _list_src=list(path_to_images + _image_teste_list_names),
            _list_thumbnail=list(path_to_thumbnails + _image_teste_list_thumbs),
            _list_name_figure=list(_image_teste_list_names),
            _list_thumbnailWidth=list(_image_teste_list_widths),
            _list_thumbnailHeight=list(_image_teste_list_heights),
            _list_isSelected= _image_teste_list_selection,
            _list_custom_data=list(_image_teste_list_custom_data),
            _list_caption=_image_teste_list_texts,
            _list_thumbnailCaption=_image_teste_list_caption,
            #_list_tags=[[{'value': "Nature", 'title': "Nature | Flowers"}]] * _image_teste_list_correct_label.shape[0])
            _list_tags=[[]] * _image_teste_list_correct_label.shape[0])

        if s_chart_flag_data == 1:
            #print('app.py chamando fpc via if')
            fig3 =  f_figure_paralelas_coordenadas(
                _df=_df,
                _filtered_df = filtered_df,
                _columns=_columns_paralelas_coordenadas,
                _selected_custom_data=json.loads(i_selected_custom_points),
                _fig = s_g_coordenadas_paralelas_figure
            )
        else:
            #print('app.py chamando fpc via else')
            fig3 =  f_figure_paralelas_coordenadas(
                        _df=_df,
                        _filtered_df = filtered_df,
                        _columns=_columns_paralelas_coordenadas,
                        _selected_custom_data=json.loads(i_selected_custom_points),
                        _fig = None
                    )

        return [fig_scatter, fig2, fig3]

    else:
        return [s_g_scatter_plot_figure, s_g_image_selector_images, s_g_coordenadas_paralelas_figure]

##############################################################################################################

opened = False
if not opened:
    webbrowser.open('http://127.0.0.1:' + str(port) + '/', new=2, autoraise=True)
    opened = False

if __name__ == '__main__':
    app.title = 'IAT'
    app.run_server(debug=False, port=port)

