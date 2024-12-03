#%%1
import os
import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
from pathlib import Path

#%%2
# Abrindo arquivos
pathDS = Path('/home/lcqar/ENS410064/2024/dados_entrada/congonhas')
files = [i for i in pathDS.rglob('*.tif')]

#%%3
# Carregandi DataArrays
xdas = [rxr.open_rasterio(file) for file in files]
xdas

# Criando índices de tempo a partir dos nomes dos arquivos
def time_index_from_filenames(filenames):
    # congonhas_YYYY-MM-DD_HH-MM-SS
    times = []
    for file in filenames:
        only_file = os.path.basename(file)
        time = pd.to_datetime(only_file.split('_')[1] + ' ' + only_file.split('_')[2].replace('-', ':')[:8], format='%Y-%m-%d %H:%M:%S')
        times.append(time)
    return pd.DatetimeIndex(times)


# Carregandi índices de tempo
time_indexes = time_index_from_filenames([file.stem for file in files])
time_indexes = xr.Variable('time', time_indexes)
time_indexes

#%%4

# Concatenando pela dimensão de tempo
ds = xr.concat(xdas, dim=time_indexes)
# Convertendo para Daset
ds = ds.to_dataset(name='color')
ds
del xdas 
x = ds.x[:]
y = ds.y[:]
#%%5

# Cores de limites
limits = {'0':'#000000',   # 0 <= v < 0.01
          '1':"#777777",   # 0.01 <= v < 0.4
          '2':'#FF2323',   # 0.4 <= v < 0.8
          '3':'#FFFF37',   # 0.8 <= v < 1
          '4':'#2BC82B'}   # v == 1

# Transformando os valores haxadecimais em rgb
r_values = [int(limit[1:3], 16) for limit in limits.values()]
g_values = [int(limit[3:5], 16) for limit in limits.values()]
b_values = [int(limit[5:8], 16) for limit in limits.values()]

# Salvando valores de RGB em dicionário com clase de classificações
rgb_values = [list(i) for i in zip(r_values,g_values,b_values)]
# rgb_values = dict(zip(limits.keys(),rgb_values))


# print(rgb_values)
# print(ds.shape)

# Deletando variáveis que já foram utilizadas
del r_values, g_values, b_values


#%%5.1
def rgba_to_class(arr: np.array) -> int:
    #print(arr)
    # Repete np.array p/ nº de vezes do rgb_values
    # Compara com rgb_values
    # Pensar que recebe uma variável arr com valor [0, 0, 0]
    #return rgb_values.index(arr) if arr in rgb_values else 0
#    return rgb_values.index(arr) if (arr[0]==rgb_values[0]) else 0
    
    #for rgb_value in rgb_values:
    #    if arr == rgb_value:
    #        return rgb_values.index(rgb_value)
    
    for i, ref in enumerate(rgb_values):
        if np.array_equal(arr, ref):
            return i
    return 0

print(rgba_to_class([255,35,35]))

#%%6

# ds.reduce(rgba_to_class, dim='band', keep_attrs =True, keepdims=True)
nds = xr.apply_ufunc(rgba_to_class, ds, input_core_dims=[['band']], vectorize=True)

#%%
testAll=[]
for classification, rgb_value in rgb_values.items():
    testband = (ds[:,0,:,:]==rgb_value[0]) & (ds[:,1,:,:]==rgb_value[1]) &\
        (ds[:,2,:,:]==rgb_value[2])
    testAll.append(testband)

del testband

#%%

testAll = np.concat(testAll)
testAll = xr.concat(np.concat(testAll).reshape(len(limits),ds.shape[0],
                ds.shape[2],ds.shape[3]), dim=time_indexes)

testAll = testAll.reshape(len(limits),ds.shape[0],
                ds.shape[2],ds.shape[3])

del classification

#%%7
# Criando DataArray
testAll = xr.DataArray(testAll, 
                          coords={'y': ds.y[:],'x': ds.x[:],'time': ds.time[:]}, 
                          dims=['class',"time","y","x"],
                          name = 'testAll')

testAll.rxr.write_crs('EPSG:4326', inplace=True)

del ds
#%%8
# Ler shapefile de ruas do open street maps da Geofabrik
import geopandas as gpd

pathSHP = "/home/lcqar/ENS410064/2024/dados_entrada/roads_sudeste/gis_osm_roads_free_1.shp"
gdf = gpd.read_file(pathSHP)

gdf

#%%9

gdf.crs

#%%
# DEFINIR AS COORDENADAS DOS LIMITES DO DATASET
x_min = testAll_xr.coords['x'].min().values
x_max = testAll_xr.coords['x'].max().values
y_min = testAll_xr.coords['y'].min().values
y_max = testAll_xr.coords['y'].max().values
print(x_min,x_max,y_min,y_max)

from shapely.geometry import box
congonhas_bounds = box(x_min,y_min,x_max,y_max)

# CORTAR AS RUAS DO OPEN STREET MAPS PARA OS LIMITES
gdf_clipped = gdf.clip(congonhas_bounds)

del gdf

#%%

gdf_clipped.plot()


#%%

testAll_xr


#gdf_clipped.crs


# DEFINIR UMA AMOSTRA DOS DADOS PARA TRABALHAR COM MENOS MEMÓRIA
# CRUZAR A LOC DAS RUAS COM AS CLASSES DE VELOCIDADE
# DIZER QUAL RUA DO OPEN STREET MAPS TEM QUAL CLASSIFICAÇÃO
