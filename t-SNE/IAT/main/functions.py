
from webbrowser import BackgroundBrowser
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from math import ceil, floor
from timeit import default_timer as timer
from os import listdir
from os.path import join

background_color = 'rgba(255, 250, 240, 100)'
aux_list = [] * 7 #new

def get_color(i):
    r = (201*i+100)%256
    g = (91*i+149)%256
    b = (53*i+237)%256
    return 'rgb(' + str(r) + ', ' + str(g) + ', ' + str(b) + ')'

def get_colorscale(max_color):
    colorscale = []
    if max_color < 1:
        max_color = 1
    inc = 1/max_color
    for i in range(max_color+1):
        colorscale.append([i*inc, get_color(i)])
    return colorscale

def compute_img_ratios(path_to_images, names):
    widths = []
    heights = []
    for p in names:
        path = join('main', path_to_images, p)
        with Image.open(path) as img:
            width, height = img.size
            r = height/width
            if r < 0.25:
                r = 0.25
            if r > 1.6:
                r = 1.6
            widths.append(1)
            heights.append(r)
    return widths, heights


def create_list_dics(
    _list_src,
    _list_thumbnail,
    _list_name_figure,
    _list_thumbnailWidth=None,
    _list_thumbnailHeight=None,
    _list_isSelected=None,
    _list_custom_data=None,
    _list_caption=None,
    _list_thumbnailCaption=None,
    _list_tags=None):

    _data = {
      'src': _list_src,
      'thumbnail': _list_thumbnail,
      'name_figure': _list_name_figure,
      'thumbnailWidth': _list_thumbnailWidth,
      'thumbnailHeight': _list_thumbnailHeight,
      'isSelected': _list_isSelected,
      'custom_data': _list_custom_data,
      'caption': _list_caption,
      'thumbnailCaption': _list_thumbnailCaption,
      'tags': _list_tags
    }

    _df_dict = pd.DataFrame(_data)
    _df_dict = _df_dict[['src', 'thumbnail', 'name_figure','thumbnailWidth','thumbnailHeight','isSelected','custom_data', 'caption', 'thumbnailCaption', 'tags']]
    _df_dict = _df_dict.to_dict('records')

    return _df_dict


def f_figure_scatter_plot(_df, _columns, _selected_custom_data, prev_fig = None, order_by = 'A-Z, a-z', 
                          background_img = 'main/assets/temp.png', xrange = [-80,80], yrange=[-80,80], opacity_changed = False, opacity=0, marker_size = 10):
    #start = timer()

    l_data = []
    column_name = 'manual_label'
    column_colors = 'colors'

    if '(binary)' in order_by:
        column_name = 'binary_label'
        column_colors = 'binary_color'

    label_names = _df[column_name].unique().tolist()

    freq_colors = []
    for l in label_names:
        freq_colors.append(len(_df[_df[column_name] == l]))

    if 'Frequency' in order_by:
        sorted_names_freq = sorted(zip(freq_colors, label_names), reverse=True)
        sorted_names = [x for _,x in sorted_names_freq]    
    else:
        sorted_names_freq = sorted(zip(label_names, freq_colors), reverse=False)
        sorted_names = [x for x,_ in sorted_names_freq]    

    for name in sorted_names:
        #_selectedpoints = _df[column_name][
        #    (_df[column_name] == idx) &
        #    (_df['custom_data'].isin(_selected_custom_data))
        #].index.values
        val = _df[_df[column_name] == name].reset_index()
        idx = int(val[column_colors].iloc[0])

        _selectedpoints = val.index[val['custom_data'].isin(_selected_custom_data)].tolist()

        sub_df = _df[column_name] == name
        _custom_points = _df['custom_data'][(sub_df)]
        _label = name + ' (' + str(val.shape[0]) + ')'

#        _temp = []
#        for i in range(len(_selectedpoints)):
#            _temp.append(val.index.get_loc(_selectedpoints[i]))
#        _selectedpoints = _temp

        scatter = go.Scattergl(
            name=_label,
            #hoverinfo='skip',
            x=val[_columns[0]],
            y=val[_columns[1]],
            selectedpoints=_selectedpoints,
            customdata=_custom_points,
            mode="markers",
            #marker=dict(size=20, symbol="circle", colorscale='rainbow'),
            marker=dict(color = get_color(idx), size=marker_size, symbol="circle"),
            hovertemplate='id = %{customdata}',
        )
        l_data.append(scatter)
    
#    scatter_time = timer()
#    print('Scatter time:', scatter_time - start) 



    layout = go.Layout(
        modebar_orientation='h',
        legend=dict(yanchor='top', y=0.9),
        xaxis={'visible': False, 'autorange': True, 'zeroline': True, 'showgrid': True, 'gridcolor': 'rgba(0,0,0,0)'},
        yaxis={'visible': False, 'autorange': True, 'zeroline': True, 'showgrid': True, 'gridcolor': 'rgba(0,0,0,0)'},
#        xaxis={'range': [-1, 1.1], 'autorange': True,
#               'gridcolor': 'rgba(0,0,0,0)', 'zeroline': False, 'showgrid': False},
#        yaxis={'range': [-1, 1.1], 'autorange': True,
#               'gridcolor': 'rgba(0,0,0,0)', 'zeroline': False, 'showgrid': False},
        plot_bgcolor=background_color,
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        dragmode='select',
        paper_bgcolor='aliceblue',
        showlegend = True
    )

    if prev_fig is not None:
        prev_data = prev_fig.data

        map_name_visibility = {}
        for d in prev_data:
            name = d['name'].split(' (')[0]
            map_name_visibility[name] = d['visible']

        for d in l_data:
            name = d['name'].split(' (')[0]
            if name in map_name_visibility:
                d['visible'] = map_name_visibility[name]
        #copy_time = timer()

        #copy_time2 = timer()
        #print('Copy time:', copy_time2 - copy_time) 
        prev_layout = prev_fig.layout
        
        layout['xaxis'] = prev_layout['xaxis']
        layout['yaxis'] = prev_layout['yaxis']
        layout['dragmode'] = prev_layout['dragmode']
        if not opacity_changed:
            layout['images'] = prev_layout['images']
    
    fig = go.Figure(data=l_data, layout=layout)

#    draw_time = timer()

    if prev_fig is None or opacity_changed:
        if opacity > 0:
            fig.add_layout_image(
            dict(
                source=Image.open(background_img),
                xref="x",
                yref="y",
                x=xrange[0],
                y=yrange[1],
                sizex=xrange[1]-xrange[0],
                sizey=yrange[1]-yrange[0],
                sizing="stretch",
                opacity=opacity,
                layer="below")
            )
#    end = timer()
#    print('Draw time:', end - draw_time) 
#    print('Total time:', end - start) 

    return fig


def f_figure_paralelas_coordenadas(_df, _filtered_df, _columns, _selected_custom_data, _fig=None):

    #print('functions.py entrou no f_figure_paralelas_coordenadas')

    #_df_temp = _df[_df['custom_data'].isin(_selected_custom_data)]
    #_df_temp = _df_temp.reset_index(drop=True)


    #_list_visible = []
    #for i in range(len(_columns)):
    #    _list_visible.append(True)


    _list_range = []
    _list_values = []

    max_color = int(_df['colors'].max())

    
    # range com max e min para cada dimensao de tsne
    for column in _columns:
        dim_min = _df[column].min()
        dim_max = _df[column].max()
        #dim_max += 0.1 # small overhead to place initial void selection
        _list_range.append([dim_min, dim_max])
        if _selected_custom_data != []:
            _list_values.append(_filtered_df[column].tolist())
        else:
            _list_values.append([dim_max+1]) #dummy point

    # determinar intervalos para selecoes de constraint range
    _list_constraint_range = []
        

    if _fig == None:
        if _selected_custom_data == []:
            for column in _columns:
                dim_max = _df[column].max()
                _list_constraint_range.append([[dim_max+50, dim_max + 55]])
                
        else:
            for column in _columns:
                col_min = _filtered_df[column].min()-0.1
                col_max = _filtered_df[column].max()+0.1
                ranges_list = [col_min, col_max]
                _list_constraint_range.append(ranges_list)

    else:
        _fig_list = _fig['data'][0]['dimensions']
        for i in range(len(_columns)):
            if 'constraintrange' in _fig_list[i]:
                _list_constraint_range.append(_fig_list[i]['constraintrange'])
            else:
                dim_max = _df[column].max()
                _list_constraint_range.append([[dim_max+50, dim_max + 55]])
                #_list_constraint_range.append([])
    
    
    _data = {
        'label': _columns,
        #'visible': _list_visible,
        'values': _list_values,
        'range': _list_range,
        'constraintrange': _list_constraint_range
    }

    dimensions_dict = pd.DataFrame(_data)
    dimensions_dict = dimensions_dict.to_dict('records')
    #for i in range(len(dimensions_dict)):
    #    dimensions_dict[i]['ticktext'] = []
        #dimensions_dict[i]['tickvals'] = []
        #dimensions_dict[i]['label'] = ''

    #print(colors)
    custom_colorscale = get_colorscale(max_color)
    #print('count: ', values, counts)

    layout = go.Layout(
#        margin={'l': 10, 'r': 10, 'b': 5, 't': 5},
        paper_bgcolor = background_color,
    )

    if max_color == 0: # fixing color when no labels are given
        max_color = 1 

    coordenadas_paralelas = go.Parcoords(
            line = dict(color = _filtered_df['colors'],
                colorscale = custom_colorscale,
                showscale = False,
                cmin=0,
                cmax=max_color
            ),
            dimensions = dimensions_dict,
            customdata = _df['custom_data']
    )
        

    
    #print('coordenadas_paralelas[dimensions]')
    #print(coordenadas_paralelas['dimensions'])
    
    l_data = []
    l_data.append(coordenadas_paralelas)
    
    figure = go.Figure(data=l_data, layout=layout)

    return figure

def init_for_update_pc(selected_points):
    global aux_list
    aux_list = []
    for _ in range(7):
        aux_list.append(selected_points.copy())


def update_df_paralelas_coord(
    _df,
    _list_columns,
    _figure,
    _new_dim_vals): # new dimension values whenever user changes selection of a specific parcoord

    global aux_list

    for i in range(len(aux_list)):
        aux = aux_list[i]

    par_coord_data = _figure['data'][0]
    curr_dims = par_coord_data.get('dimensions', None)

    parcoord_index = int( list( _new_dim_vals[0].keys())[0] . split('.')[0][-2:-1] )
    i = parcoord_index
    points_in_i = []
    _vals = _df[_list_columns[i]].tolist()
    _data = _df['custom_data'].tolist()

    if 'constraintrange' in curr_dims[i]:
        if isinstance(curr_dims[i]['constraintrange'][0], list):
            for _ranges in curr_dims[i]['constraintrange']:
                for j in range(len(_vals)):
                    val = _vals[j]
                    if val >= _ranges[0] and val <= _ranges[1]:
                        points_in_i.append(_data[j])
            aux_list[i] = points_in_i

        else:
            _ranges = curr_dims[i]['constraintrange']
            for j in range(len(_vals)):
                    val = _vals[j]
                    if val >= _ranges[0] and val <= _ranges[1]:
                        points_in_i.append(_data[j])

                #aux_list[i] = updated_df.query(query_string)
            aux_list[i] = points_in_i
    else:
        aux_list[i] = []
                
    intersected_points = aux_list[0]
    for i in range(len(aux_list)):
        aux = aux_list[i]
        if len(aux) == 0:
            return []
        elif len(aux) < len(intersected_points):
            intersected_points = aux

    for aux in aux_list:
        intersected_points = np.intersect1d(intersected_points, aux)

    return list(intersected_points)
    #return pd.DataFrame(columns=updated_df.columns)

def get_image(path, paint = False, color = (1, 1, 1), zoom=0.2, dim = 255):
    img = Image.open(path).convert('RGBA')
    img = np.array(img)
    if paint:
        img[:,:,0] = np.uint8(img[:,:,0] * color[0])
        img[:,:,1] = np.uint8(img[:,:,1] * color[1])
        img[:,:,2] = np.uint8(img[:,:,2] * color[2])
        img[:,:,3] = dim
    img = Image.fromarray(img)
    
    return OffsetImage(img, zoom=zoom)

def map_of_images(df, fig_scatter, path_to_images):
    output_path = 'main/assets/temp.png'

    fig_scatter = go.Figure(fig_scatter)
    fig_layout = fig_scatter.layout
    xrange = [floor(fig_layout['xaxis'][0]), ceil(fig_layout['xaxis'][1])]
    yrange = [floor(fig_layout['yaxis'][0]), ceil(fig_layout['yaxis'][1])]

    df_filtered = df[(df['x'] >= xrange[0]) & (df['x'] <= xrange[1]) & (df['y'] >= yrange[0]) & (df['y'] <= yrange[1])]
    
    x = df_filtered['x']
    y = df_filtered['y']
    names = df_filtered['names']
    paths = ['main/' + path_to_images + n for n in names]
    zoom = 12/(xrange[1]-xrange[0])

    f = plt.figure(figsize=(24,24), frameon=False)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.axis('off')
    f.add_axes(ax)
    ax.scatter(x, y, s=0) 

    for xs, ys, path in zip(x, y,paths):
        ab = AnnotationBbox(get_image(path, zoom=zoom), (xs, ys), frameon=False, box_alignment=(1, 1))
        ax.add_artist(ab)
        
    #plt.grid()
    #plt.axis('off')

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    f.savefig(output_path, bbox_inches='tight', pad_inches = 0)

    #image = Image.open(output_path)
    #new_image = Image.new("RGBA", image.size, "WHITE")
    #new_image.paste(image, (0, 0), image)              
    #new_image.convert('RGB').save(output_path, "PNG") 

    return output_path
