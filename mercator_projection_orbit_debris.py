from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colorbar import ColorbarBase
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.
dfa = pd.read_csv('latlong_data_4weeks_testing.csv')
dfa5_6 = dfa.loc[(dfa['2'] >= 500) & (dfa['2'] <= 600)]
dfa6_7 = dfa.loc[(dfa['2'] >= 600) & (dfa['2'] <= 700)]
dfa7_8 = dfa.loc[(dfa['2'] >= 700) & (dfa['2'] <= 800)]
dfa8_9 = dfa.loc[(dfa['2'] >= 800) & (dfa['2'] <= 900)]
dfa9_10 = dfa.loc[(dfa['2'] >= 900) & (dfa['2'] <= 1000)]
dfa10_11 = dfa.loc[(dfa['2'] >= 1000) & (dfa['2'] <= 1100)]
dfa11_12 = dfa.loc[(dfa['2'] >= 1100) & (dfa['2'] <= 1200)]
dfa12_13 = dfa.loc[(dfa['2'] >= 1200) & (dfa['2'] <= 1300)]
dfa13_14 = dfa.loc[(dfa['2'] >= 1300) & (dfa['2'] <= 1400)]

lat_resolution = 150



for lon_idx in range(1,int((360/15)+1)): # 1 to 24
    # first nest of loop, goes through each column in map
    current_lon_list = []
    for lat_idx in range(1,int((160/10)+1)): # 1 to 16
        current_min_lat = -80 + (lat_idx-1)*(160/16)
        current_max_lat = -80 + (lat_idx)*(160/16)
        current_min_lon = -180 + (lon_idx-1)*(360/24)
        current_max_lon = -180 + (lon_idx)*(360/24)


        histQuant_df = dfa13_14.loc[(dfa['0'] >= current_min_lat) & (dfa['0'] <= current_max_lat) & (dfa['1'] >= current_min_lon) & (dfa['1'] <= current_max_lon)]
        histQuant = histQuant_df.shape[0]

        if lat_idx==1 or lat_idx==16:
            quantLength = int(round(lat_resolution*(113 / 779)))
        elif lat_idx==2 or lat_idx==15:
            quantLength = int(round(lat_resolution*(67 / 779)))
        elif lat_idx==3 or lat_idx==14:
            quantLength = int(round(lat_resolution * (48 / 779)))
        elif lat_idx==4 or lat_idx==13:
            quantLength = int(round(lat_resolution * (40 / 779)))
        elif lat_idx==5 or lat_idx==12:
            quantLength = int(round(lat_resolution * (34 / 779)))
        elif lat_idx==6 or lat_idx==11:
            quantLength = int(round(lat_resolution * (31 / 779)))
        elif lat_idx==7 or lat_idx==10:
            quantLength = int(round(lat_resolution * (27 / 779)))
        elif lat_idx==8 or lat_idx==9:
            quantLength = int(round(lat_resolution * (26 / 779)))

        for k in range(1,quantLength+1):
            current_lon_list.append(histQuant)

    # after latitude loop
    #for p in range(1,5):
        #current_lon_list.append(histQuant) # adds last known histQuant value(top of map) to match resolution
    #current_lon_list = current_lon_list[:len(current_lon_list) - n]


    new_col = np.transpose(np.array([current_lon_list]))

    if lon_idx==1:
        updatedMatrix = new_col
    else:
        updatedMatrix = np.hstack((updatedMatrix,new_col))


print(updatedMatrix)
print(updatedMatrix.shape)

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
ax = plt.gca()
fig = plt.gcf()

#data = np.random.rand(150, 24) * 15
#print(data)
cmap = colors.ListedColormap(['#000080', '#5EA5FF', '#40EBFF', '#00FFFF', '#2BFF97', '#2EFF3B', '#00FF00',
                              '#51FF17', '#AFFF12', '#FFFF00', '#FFE810', '#FFBD12', '#FF8000', '#C03C1D', '#A2052D', '#FF0000']) # get custom colors from
# matlab rgb values, convert to hex
bounds = [0,4,8,12,17,21,24,29,33,37,41,45,50,54,58,62]
norm = colors.BoundaryNorm(bounds, cmap.N)
m.imshow(updatedMatrix, cmap=cmap, norm=norm, alpha=0.5)

m.drawcoastlines()


#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.

m.drawparallels(np.arange(-90.,91.,10.), labels=[1,0,0,0], labelstyle='+/-')
m.drawmeridians(np.arange(-180.,181.,15.), labels=[0,0,0,1], labelstyle='+/-')
#m.drawmapboundary(fill_color='aqua')

plt.title("Mercator Projection, 1300-1400km")
cax = fig.add_axes([0.8, 0.15, 0.04, 0.7]) # posititon
cb = ColorbarBase(cax,cmap=cmap,norm=norm, orientation='vertical')
plt.show()

