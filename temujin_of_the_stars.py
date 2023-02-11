import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


def sat_data2df(filename):
    data = pd.read_csv(filename, converters={'Unnamed: 4': str})
    data = data.iloc[1:]
    data = data.iloc[:, :-1]

    data.columns = ['Line element #', 'Satellite Number', 'Inclination [deg]',
                    'RAAN [deg]', 'Eccentricity (leading decimal point assumed)',
                    'Argument of Perigee [deg]', 'Mean Anomaly [deg]', 'Last 3 cols']

    # last 3 cols = Mean Motion [revs per day](53-63), Revolution number at epoch [revs](64-68), Checksum (Modulo 10)(69)

    line2_elements_sat = data.iloc[::3, :]

    return line2_elements_sat


line2_elements1 = sat_data2df('FENGYUN_1C_data.csv')
line2_elements2 = sat_data2df('IRIDIUM_33_data.csv')
line2_elements3 = sat_data2df('COSMOS_2251_data.csv')
line2_elements4 = sat_data2df('COSMOS_1408_data.csv')


line2_elements5 = sat_data2df('FENGYUN_1C_data2.csv')
line2_elements6 = sat_data2df('IRIDIUM_33_data2.csv')
line2_elements7 = sat_data2df('COSMOS_2251_data2.csv')
line2_elements8 = sat_data2df('COSMOS_1408_data2.csv')

line2_elements9 = sat_data2df('FENGYUN_1C_data3.csv')
line2_elements10 = sat_data2df('IRIDIUM_33_data3.csv')
line2_elements11 = sat_data2df('COSMOS_2251_data3.csv')
line2_elements12 = sat_data2df('COSMOS_1408_data3.csv')

line2_elements13 = sat_data2df('FENGYUN_1C_data4.csv')
line2_elements14 = sat_data2df('IRIDIUM_33_data4.csv')
line2_elements15 = sat_data2df('COSMOS_2251_data4.csv')
line2_elements16 = sat_data2df('COSMOS_1408_data4.csv')


line2_elements = pd.concat([line2_elements1, line2_elements2, line2_elements3, line2_elements4,
                            line2_elements5, line2_elements6, line2_elements7, line2_elements8,
                            line2_elements9, line2_elements10, line2_elements11, line2_elements12,
                            line2_elements13, line2_elements14, line2_elements15, line2_elements16], axis=0) # either add lat long csvs together by row,
                                                                                                         # or just process all data files from over a period of time over weeks,
                                                                                                         # ex: 16 line2_elements for 4 weeks processed all at once into one final csv


#line2_elements = pd.concat([line2_elements1, line2_elements2, line2_elements3, line2_elements4], axis=0)
print(line2_elements)




def orbit_info2latlong(inclination, RAAN, arg_per, eccen, mean_anomaly, mean_motion):

    i = np.deg2rad(inclination)
    omega_l = np.deg2rad(arg_per)
    omega_b = np.deg2rad(RAAN)
    mean_anomaly_rad = np.deg2rad(mean_anomaly)
    mu = 3.986E5 # km^3/s^2
    radius_earth = 6371 # km
    e = eccen

    E_initial_guess = mean_anomaly_rad

    func1 = lambda eccen_anom_rad : mean_anomaly_rad - eccen_anom_rad + eccen*np.sin(eccen_anom_rad)

    E_rad = fsolve(func1, E_initial_guess)[0]
    f_rad = 2*np.arctan( np.sqrt((1+e) / (1-e))*np.tan(E_rad/2))
    print('Eccentric anomaly [rad], E:', E_rad)
    print('e:', e)
    print('f_rad:',f_rad)
    print('omega_l:',omega_l)
    print('omega_l + f_rad:', omega_l + f_rad)
    print('np.sin(omega_l + f_rad):', np.sin(omega_l + f_rad))
    print('--------------------------------')
    f_rad = 0 # fix f to 0 like before to check if issue

    period = 2*np.pi*(mean_motion * ( (2*np.pi) / (24*3600)))**(-1)


    semi_major_axis = ( (period**2 / (4 * np.pi**2) ) * (mu) )**(1/3)


    radius_act = semi_major_axis*(1 - e*np.cos(E_rad))
    radius_avg = semi_major_axis*(1 + 0.5*e**2)


    altitude = radius_avg - radius_earth


    phi_lat = np.rad2deg( np.arcsin(np.sin(i) * np.sin(omega_l + f_rad)) )

    lambda_long = np.rad2deg( omega_b + np.arctan(np.cos(i) * np.tan(omega_l + f_rad)))

    return (phi_lat, lambda_long, altitude) # deg, deg, km

results_list = []


for index, row in line2_elements.iterrows():
   #print("0." + str(row['Eccentricity (leading decimal point assumed)']))
    e = float("0." + str(row['Eccentricity (leading decimal point assumed)']))
    mean_anomaly = float(row['Mean Anomaly [deg]'])
    mean_motion = float(row['Last 3 cols'])
    sol = orbit_info2latlong(float(row['Inclination [deg]']), float(row['RAAN [deg]']), float(row['Argument of Perigee [deg]']), e, mean_anomaly, mean_motion)
    results_list.append(sol) # list of lat, long, and altitude pairs

adapted_list = []


for i in results_list:

    if i[1] > 360:
        new_lambda = i[1] - 360
    else:
        new_lambda = i[1]

    if new_lambda > 180:
        new_lambda = new_lambda - 360


    adapted_list.append((i[0], new_lambda, i[2]))



#for o in adapted_list:
    #print(o)
dfa = pd.DataFrame(adapted_list)
dfa.to_csv("latlong_data_4weeks_testing.csv", sep=',')

latAmplitudes5_6 = []
longAmplitudes5_6 = []

latAmplitudes6_7 = []
longAmplitudes6_7 = []

latAmplitudes7_8 = []
longAmplitudes7_8 = []

latAmplitudes8_9 = []
longAmplitudes8_9 = []

latAmplitudes9_10 = []
longAmplitudes9_10 = []

latAmplitudes10_11 = []
longAmplitudes10_11 = []

latAmplitudes11_12 = []
longAmplitudes11_12 = []

latAmplitudes12_13 = []
longAmplitudes12_13 = []

latAmplitudes13_14 = []
longAmplitudes13_14 = []


for k in adapted_list:
    if  (500 <= k[2] < 600):
        latAmplitudes5_6.append(k[0])
    elif (600 <= k[2] < 700):
        latAmplitudes6_7.append(k[0])
    elif (700 <= k[2] < 800):
        latAmplitudes7_8.append(k[0])
    elif (800 <= k[2] < 900):
        latAmplitudes8_9.append(k[0])
    elif (900 <= k[2] < 1000):
        latAmplitudes9_10.append(k[0])
    elif (1000 <= k[2] < 1100):
        latAmplitudes10_11.append(k[0])
    elif (1100 <= k[2] < 1200):
        latAmplitudes11_12.append(k[0])
    elif (1200 <= k[2] < 1300):
        latAmplitudes12_13.append(k[0])
    elif (1300 <= k[2] < 1400):
        latAmplitudes13_14.append(k[0])

for k in adapted_list:
    if (500 <= k[2] < 600):
        longAmplitudes5_6.append(k[1])
    elif (600 <= k[2] < 700):
        longAmplitudes6_7.append(k[1])
    elif (700 <= k[2] < 800):
        longAmplitudes7_8.append(k[1])
    elif (800 <= k[2] < 900):
        longAmplitudes8_9.append(k[1])
    elif (900 <= k[2] < 1000):
        longAmplitudes9_10.append(k[1])
    elif (1000 <= k[2] < 1100):
        longAmplitudes10_11.append(k[1])
    elif (1100 <= k[2] < 1200):
        longAmplitudes11_12.append(k[1])
    elif (1200 <= k[2] < 1300):
        longAmplitudes12_13.append(k[1])
    elif (1300 <= k[2] < 1400):
        longAmplitudes13_14.append(k[1])


def make_3dhistvals(xamp, yamp):
    x = np.array(xamp)
    y = np.array(yamp)

    hist, xedges, yedges = np.histogram2d(x, y, bins=(10, 10))
    xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

    xpos = xpos.flatten() / 2.
    ypos = ypos.flatten() / 2.
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    return xpos, ypos, zpos, dx, dy, dz, rgba

hist1 = make_3dhistvals(latAmplitudes5_6, longAmplitudes5_6)
hist2 = make_3dhistvals(latAmplitudes6_7, longAmplitudes6_7)
hist3 = make_3dhistvals(latAmplitudes7_8, longAmplitudes7_8)
hist4 = make_3dhistvals(latAmplitudes8_9, longAmplitudes8_9)
hist5 = make_3dhistvals(latAmplitudes9_10, longAmplitudes9_10)
hist6 = make_3dhistvals(latAmplitudes10_11, longAmplitudes10_11)
hist7 = make_3dhistvals(latAmplitudes11_12, longAmplitudes11_12)
hist8 = make_3dhistvals(latAmplitudes12_13, longAmplitudes12_13)
hist9 = make_3dhistvals(latAmplitudes13_14, longAmplitudes13_14)

fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax1 = fig.add_subplot(331, projection='3d')
ax2 = fig.add_subplot(332, projection='3d')
ax3 = fig.add_subplot(333, projection='3d')
ax4 = fig.add_subplot(334, projection='3d')
ax5 = fig.add_subplot(335, projection='3d')
ax6 = fig.add_subplot(336, projection='3d')
ax7 = fig.add_subplot(337, projection='3d')
ax8 = fig.add_subplot(338, projection='3d')
ax9 = fig.add_subplot(339, projection='3d')

alpha_global = 0.7
ax1.bar3d(hist1[0], hist1[1], hist1[2], hist1[3], hist1[4], hist1[5], color=hist1[6], zsort='average', alpha=alpha_global)
ax2.bar3d(hist2[0], hist2[1], hist2[2], hist2[3], hist2[4], hist2[5], color=hist2[6], zsort='average', alpha=alpha_global)
ax3.bar3d(hist3[0], hist3[1], hist3[2], hist3[3], hist3[4], hist3[5], color=hist3[6], zsort='average', alpha=alpha_global)
ax4.bar3d(hist4[0], hist4[1], hist4[2], hist4[3], hist4[4], hist4[5], color=hist4[6], zsort='average', alpha=alpha_global)
ax5.bar3d(hist5[0], hist5[1], hist5[2], hist5[3], hist5[4], hist5[5], color=hist5[6], zsort='average', alpha=alpha_global)
ax6.bar3d(hist6[0], hist6[1], hist6[2], hist6[3], hist6[4], hist6[5], color=hist6[6], zsort='average', alpha=alpha_global)
ax7.bar3d(hist7[0], hist7[1], hist7[2], hist7[3], hist7[4], hist7[5], color=hist7[6], zsort='average', alpha=alpha_global)
ax8.bar3d(hist8[0], hist8[1], hist8[2], hist8[3], hist8[4], hist8[5], color=hist8[6], zsort='average', alpha=alpha_global)
ax9.bar3d(hist9[0], hist9[1], hist9[2], hist9[3], hist9[4], hist9[5], color=hist9[6], zsort='average', alpha=alpha_global)

ax1.set_title("500-600km altitude")
ax2.set_title("600-700km altitude")
ax3.set_title("700-800km altitude")
ax4.set_title("800-900km altitude")
ax5.set_title("900-1000km altitude")
ax6.set_title("1000-1100km altitude")
ax7.set_title("1100-1200km altitude")
ax8.set_title("1200-1300km altitude")
ax9.set_title("1300-1400km altitude")

ax1.set_xlabel("Latitude [deg]")
ax2.set_xlabel("Latitude [deg]")
ax3.set_xlabel("Latitude [deg]")
ax4.set_xlabel("Latitude [deg]")
ax5.set_xlabel("Latitude [deg]")
ax6.set_xlabel("Latitude [deg]")
ax7.set_xlabel("Latitude [deg]")
ax8.set_xlabel("Latitude [deg]")
ax9.set_xlabel("Latitude [deg]")

ax1.set_ylabel("Longitude [deg]")
ax2.set_ylabel("Longitude [deg]")
ax3.set_ylabel("Longitude [deg]")
ax4.set_ylabel("Longitude [deg]")
ax5.set_ylabel("Longitude [deg]")
ax6.set_ylabel("Longitude [deg]")
ax7.set_ylabel("Longitude [deg]")
ax8.set_ylabel("Longitude [deg]")
ax9.set_ylabel("Longitude [deg]")


plt.savefig("Debris visualizer")
#plt.show()

lat_resolution = 300
print(dfa)
dfa5_6 = dfa.loc[(dfa[2] >= 500) & (dfa[2] <= 600)]


for lon_idx in range(1,int((360/15)+1)): # 1 to 24
    # first nest of loop, goes through each column in map
    current_lon_list = []
    for lat_idx in range(1,int((160/10)+1)): # 1 to 16
        current_min_lat = -80 + (lat_idx-1)*(160/10)
        current_max_lat = -80 + (lat_idx)*(160/10)
        current_min_lon = -180 + (lon_idx-1)*(360/15)
        current_max_lon = -180 + (lon_idx)*(360/15)


        histQuant_df = dfa5_6.loc[(dfa[0] >= current_min_lat) & (dfa[0] <= current_max_lat) & (dfa[1] >= current_min_lon) & (dfa[1] <= current_max_lon)]
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
    for p in range(1,5):
        current_lon_list.append(histQuant) # adds last known histQuant value(top of map) to match resolution

    new_col = np.transpose(np.array([current_lon_list]))

    if lon_idx==1:
        updatedMatrix = new_col
    else:
        updatedMatrix = np.hstack((updatedMatrix,new_col))


print(updatedMatrix)
print(updatedMatrix.shape)

#print(np.transpose(np.array([current_lon_list])))


