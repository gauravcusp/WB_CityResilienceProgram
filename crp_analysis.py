import osmnx as ox
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import datetime
import numpy as np
import geocoder
from shapely.ops import cascaded_union
import geopandas as gpd
from glob import glob
import os
import pylab as pl
ox.config(log_console=True, use_cache=True)

######################################################################################################################

def get_time(t1, t2):
    
    diff = t2 - t1
    
    c = round(diff.total_seconds() / 60, 2)
    
    return c

######################################################################################################################

def get_polygon(string):
    
    lis = glob('data/AOI/*/*.shp')
    
    city = []
    for i in lis:
        x = i.split('\\')[-1]
        city.append(x.split('_')[0])
        
    if string in city:
        l = glob('data/AOI/{}/*.shp'.format(string))
        adm = gpd.GeoDataFrame.from_file(l[0])
        adm = adm.to_crs(epsg=4326)
        pol = [i for i in adm.geometry]
        boundary_poly = cascaded_union(pol)
        
    else:
        boundary_GDF = ox.gdf_from_place('{}'.format(string),which_result=1)
        boundary_poly = boundary_GDF.loc[0,'geometry']
        if boundary_poly.geom_type == 'Polygon':
            boundary_poly = boundary_poly
        else:
            try:
                boundary_GDF = ox.gdf_from_place('{}'.format(string),which_result=2)
                boundary_poly = boundary_GDF.loc[0,'geometry']
            except:
                print('Polygon not available')
                boundary_poly = -1
    
    return boundary_poly

######################################################################################################################

def get_graph(place):
    string = place.split(',')[0]
    
    print('Fetching graph data for {}'.format(place))

    poly = get_polygon(string)
    
    if poly == -1:
        gdf = ox.gdf_from_place('{}'.format(string))
        G = ox.graph_from_bbox(gdf.bbox_north, gdf.bbox_south, gdf.bbox_east, gdf.bbox_west)
        val = 0
    else:
        try:
            G = nx.read_gpickle('data/{a}/{a}'.format(a=string))
            val = 1
        except FileNotFoundError:
            G = ox.graph_from_polygon(poly, network_type='drive')
            val = 0

    
    G = ox.project_graph(G)
    
    print('Writing graph file')
    
    try:
        os.mkdir('data/{}'.format(string))
    except FileExistsError:
        pass
    
    if val != 1:
        nx.write_gpickle(G, path='data/{a}/{a}'.format(a=string))
    
    return G

######################################################################################################################


def get_centrality_stats(place):
    import numpy as np
    
    string = place.split(',')[0]
    
    try:
        edges = gpd.read_file("data/{}/edges/edges.shp".format(string))
        
        if 'edge_centr' in edges.columns:
            df = pd.DataFrame()
            df['edge_centr'] = edges.edge_centr.astype(float)
            df['edge_centr_avg'] = np.nansum(df.edge_centr.values)/len(df.edge_centr)
            df.to_csv("data/{a}/Extended_stats_{a}.csv".format(a=string))
    except FileNotFoundError:
        print("Edges file doesn't exist. Running edge_centrality function.")
        G = get_graph(G)
        extended_stats = ox.extended_stats(G, bc=True)
        dat = pd.DataFrame.from_dict(extended_stats)
        dat.to_csv('data/{a}/Extended_Stats_{b}.csv'.format(a=string, b=string))
    except Exception as e:
        print('Exception Occurred', e)
        
        
######################################################################################################################


def get_edge_centrality(place):
    
    t1 = datetime.datetime.now()
    
    string = place.split(',')[0]
    
    # download and project a street network
    
    G = get_graph(place)
    #G = ox.graph_from_place('Davao City, Philippines')
    t2 = datetime.datetime.now()
    
    print('{} minutes elapsed!'.format(get_time(t1, t2)))
    
    print('Getting node centrality')
    node_centrality = nx.betweenness_centrality(G)
    
    
    t3 = datetime.datetime.now()
    
    print('{} minutes elapsed!'.format(get_time(t1, t3)))
    
    print('Getting edge centrality')
    # edge closeness centrality: convert graph to a line graph so edges become nodes and vice versa
    edge_centrality = nx.edge_betweenness_centrality(G)
    
    t4 = datetime.datetime.now()
    
    print('{} minutes elapsed!'.format(get_time(t1, t4)))
    
    new_edge_centrality = {}

    for u,v in edge_centrality:
        new_edge_centrality[(u,v,0)] = edge_centrality[u,v]
    
    print('Saving output gdf')
    
    nx.set_node_attributes(G, node_centrality, 'node_centrality')
    nx.set_edge_attributes(G, new_edge_centrality, 'edge_centrality')
    ox.save_graph_shapefile(G, filename='{}'.format(string))
    
    t5 = datetime.datetime.now()
    
    print('{} minutes elapsed!'.format(get_time(t1, t5)))
    
    print('Getting basic stats')
    
    basic_stats = ox.basic_stats(G)
    dat = pd.DataFrame.from_dict(basic_stats)
    dat.to_csv('data/{a}/Basic_Stats_{b}.csv'.format(a=string, b=string))
    
    t6 = datetime.datetime.now()
    
    print('{} minutes elapsed!'.format(get_time(t1, t6)))
    
    print('Getting extended stats')
    
    #extended_stats = ox.extended_stats(G, bc=True)
    
    get_centrality_stats(string)
    
    #dat = pd.DataFrame.from_dict(extended_stats)
    #dat.to_csv('data/{a}/Extended_Stats_{b}.csv'.format(a=string, b=string))
    
    t7 = datetime.datetime.now()
    print('Completed with total time of {} minutes'.format(get_time(t1, t6)))
    
    return


######################################################################################################################

def get_bc_graph_plots(place):
    
    string = place.split(',')[0]
    
    G = nx.read_gpickle("data/{a}/{b}".format(a=string, b=string))
    b = ox.basic_stats(G)
    
    #G_projected = ox.project_graph(G)
    node_lis = glob('data/{}/nodes/nodes.shp'.format(string))
    extended_path_lis = glob('data/{}/Extended_*.csv'.format(string))
    
    gdf_node = gpd.GeoDataFrame.from_file(node_lis[0])
    exten = pd.read_csv(extended_path_lis[0])
    exten= exten.rename(columns={'Unnamed: 0':'osmid'})
    exten['betweenness_centrality'] = exten['edge_centr']*100
    
    max_node = exten[exten.betweenness_centrality == max(exten.betweenness_centrality)]['osmid'].values[0]
    max_bc = max(exten.betweenness_centrality)
    
    nc = ['r' if node==max_node else '#336699' for node in G.nodes()]
    ns = [80 if node==max_node else 8 for node in G.nodes()]
    
    print('{}: The most critical node has {:.2f}% of shortest journeys passing through it. \n'.format(place, max_bc))
    print('The road network of {} has {} nodes and {} edges \n\n'.format(string, b['n'], b['m']))
    fig, ax =  ox.plot_graph(G, node_size=ns, node_color=nc, node_zorder=2, node_alpha=0.8, edge_alpha=0.8,
                            fig_height=8,fig_width=8)
    gdf_node[gdf_node.osmid ==  max_node].plot(ax=ax, color='red', zorder = 3)
    
    #ax.set_title('{}: {:.2f}% of shortest paths between all nodes \n in the network through this node'.format(string, max_bc), fontsize=15)
    
    

    print('\n\n\n')
    
    fig.savefig('data/{}/{}_bc_graph_plot.png'.format(string, string), dpi=300)
    
    return


######################################################################################################################


def get_netowrk_plots(city):
    string = city.split(',')[0]
    
    G = get_graph(string)
    
    fig, ax = ox.plot_graph(G, node_color = '#336699', node_zorder = 2, node_size=5 )
    
    fig.savefig('data/{}/{}_network_plot.png'.format(string, string), dpi=300)
    
    return 

######################################################################################################################


def get_sqml_plot(place, point, network_type='drive', bldg_color='orange', dpi=300,
              dist=1200, default_width=4, street_widths=None):
    gdf = ox.footprints.footprints_from_point(point=point, distance=dist)
    fig, ax = ox.plot_figure_ground(point=point, dist=dist, network_type=network_type, default_width=default_width,
                                    street_widths=street_widths, save=False, show=False, close=True)
    fig, ax = ox.footprints.plot_footprints(gdf, fig=fig, ax=ax, color=bldg_color, set_bounds=False,
                                save=True, show=False, close=True, filename=place, dpi=dpi)
    
    
######################################################################################################################


def plot_radar(city):
    string = city.split(',')[0]
    #G = ox.graph_from_place(city, network_type='drive')
    G = get_graph(city)
    G = ox.add_edge_bearings(G)
    bearings = pd.Series([data['bearing'] for u, v, k, data in G.edges(keys=True, data=True)])
    #ax = bearings.hist(bins=30, zorder=2, alpha=0.8)
    #xlim = ax.set_xlim(0, 360)
    #ax.set_title('{} street network compass bearings'.format(city))
    #plt.show()
    fig = pl.figure()
    n = 30
    bearings = bearings[bearings>0]
    count, division = np.histogram(bearings, bins=[ang*360/n for ang in range(0,n+1)])
    division = division[0:-1]
    width =  2 * np.pi/n
    fig = pl.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    ax = pl.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    bars = ax.bar(division * np.pi/180 - width * 0.5 , count, width=width, bottom=0.0)
    ax.set_title('{} street network compass bearings'.format(city), y=1.07)
    fig.savefig('data/{a}/{a}_radar_plot.png'.format(a=string), dpi=300)
    
    return

######################################################################################################################

def get_crp_cities_stats(cities):
    """
    Input:
    
    cities: list of cities
    
    Output:
    CSV file with CRP stats for cities provided as input
    """
    
    dat = pd.DataFrame(columns = ['Number of Intersections', 'Number of Roads', 'Total Length of network (in km)',
                             'Maximum Betweenness Centrality', 'Average Betweenness Centrality'])
    num_intersect, num_roads, tot_len, max_bw, avg_bw, city = [], [], [], [], [], []
    
    for i in cities:
        bas = pd.read_csv("data/{a}/Basic_Stats_{a}.csv".format(a=i))
        ext = pd.read_csv("data/{a}/Extended_stats_{a}.csv".format(a=i))
        num_intersect.append(bas['n'].unique()[0])
        num_roads.append(bas['m'].unique()[0])
        tot_len.append(float(bas.edge_length_total.unique()[0]) / 1000)
        max_bw.append(ext.betweenness_centrality.max())
        avg_bw.append(ext.betweenness_centrality_avg.iloc[0])
        city.append(i)

    dat['Number of Intersections'] = num_intersect
    dat['Number of Roads'] = num_roads
    dat['Total Length of network (in km)'] = tot_len
    dat['Maximum Betweenness Centrality'] = max_bw
    dat['Average Betweenness Centrality'] = avg_bw
    dat.index = city

    dat.to_csv('data/CRP_Stats.csv')
    
###################################################################################################################### 


def main(city):
    
    get_edge_centrality(city)
    get_bc_graph_plots(city)
    get_netowrk_plots(city)
    plot_radar(city)
    
    dic = {'Banjul' : (13.455837, -16.575649), 'Yangon' : (16.800215, 96.149115), 'Kampala' : (0.314985, 32.586403),
          'Monrovia' : (6.301837, -10.795610), 'Kinshasa' : (-4.334033, 15.303702), 'Cotonou' : (6.369869, 2.446943), 
          'Bamako' : (12.646760, -7.993091), 'Zanzibar' : (-6.165207, 39.201966), 'Maiduguri' : (11.846616, 13.157458),
          'Port Harcourt' : (4.797717, 6.979766), 'Warri' : (5.568975, 5.788519), 'Benin' : (6.332887, 5.622240)}
    
    get_sqml_plot(city, dic[city])
    

    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
