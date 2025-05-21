


import warnings
import sys
from my_subscript import BuildParams, BuildEPL

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm
from my_subscript import BuildParams
import warnings
import sys
import traceback
from geopy.distance import geodesic
import matplotlib.gridspec as gridspec







def DefineScenario(ScenarioFolder,Params,SensorsOnly):


    import pandas as pd
    import math


    # ScenarioFolder = "C:\Drones\Python\Scenarios\\"+Scenario+"\\"
    FileName="Scenario_Definition.xlsx"

    Sensors=pd.read_excel(ScenarioFolder+FileName,'Sensors')
    if not SensorsOnly:
        WayPoints = pd.read_excel(ScenarioFolder + FileName, 'WayPoints')



    if not SensorsOnly:
        return (Sensors,WayPoints)
    else:
        return (Sensors)



def BuildScenario(WayPoints,Params):

    import math
    import navpy
    import numpy as np
    from numpy import linalg as LA
    import pymap3d as pm
    from my_subscript import calc_bearing, calc_distance,getParam,ECEF2Local_NWU,Local_NWU2ECEF, Calc_dAlpha

    T_Start=0
    dT=getParam(Params, 'AT_Sys_Params.dT')
    a=getParam(Params, 'AT_Sys_Params.a')
    e2=getParam(Params, 'AT_Sys_Params.e2')

    Ref_Point_Lat = getParam(Params, 'AT_Sys_Params.Ref_Point_Lat')
    Ref_Point_Long = getParam(Params, 'AT_Sys_Params.Ref_Point_Long')
    Ref_Point_Alt = getParam(Params, 'AT_Sys_Params.Ref_Point_Alt')



    White_Picture=pd.DataFrame({
        'ID': [],
        'Time': [],
        'Leg': [],
        'Lat': [],
        'Long': [],
        'Alt': [],
        'Speed': [],
        'Dist_To_Dest': [],
        'bearing': [],
        'Pos_x':[],
        'Pos_y': [],
        'Pos_z': [],
        'Vel_x':[],
        'Vel_y': [],
        'Vel_z': [],
        'Pos_Ref': [],
        'Vel_Ref': []
    })

    IDs = WayPoints['ID'].unique().tolist()
    for sgid in IDs:
        time = 0
        ThisWayPoints = WayPoints[WayPoints['ID'] == sgid].copy()
        # for item row in WayPoints:F
        leg=0
        for index, row in ThisWayPoints.iterrows():
            leg=leg+1
            ind=leg-1
            print ('Building leg'+str(leg), flush=True)
            # tbd handle multi targets


            # at=float(ThisWayPoints.Lat[i
            # tem-1])
            if leg==1:
                ID=ThisWayPoints.ID.iloc[ind]
                lat=ThisWayPoints.Lat.iloc[ind]
                long =ThisWayPoints.Long.iloc[ind]
                alt=ThisWayPoints.Alt.iloc[ind]
                speed=ThisWayPoints.Speed.iloc[ind]
                V_up = ThisWayPoints.VerticalSpeed.iloc[ind]
                x, y, z = pm.geodetic2ecef(lat, long, alt)
                ecef_pos = [x, y, z]
                nwu_pos = ECEF2Local_NWU(ecef_pos, lat, long, alt, a, e2, 0)
                next_lat = ThisWayPoints.Lat.iloc[ind + 1]
                next_long = ThisWayPoints.Long.iloc[ind + 1]
                bearing = calc_bearing(lat, long, next_lat, next_long)

            if ind>=len(ThisWayPoints)-1:
                break


            Hor_Acc=float(ThisWayPoints.VerticalSpeed[index])


            # enu=list(pm.ecef2enu(x, y, z, lat, long,alt))
            # tbd not in last itteration !!!!!
            next_lat = ThisWayPoints.Lat.iloc[ind + 1]
            next_long = ThisWayPoints.Long.iloc[ind + 1]
            next_speed = float(ThisWayPoints.Speed.iloc[ind + 1])
            next_alt = float(ThisWayPoints.Alt.iloc[ind + 1])

            Turn_Acc=9.8*float(ThisWayPoints.TurnAcc.iloc[ind+1])


            Dist_To_Dest = calc_distance(lat, long, next_lat, next_long)   # km
            Prev_Dist_To_Dest=Dist_To_Dest




            while (True):



                x, y, z = pm.geodetic2ecef(lat, long, alt)
                ecef_pos = [x, y, z]

                nwu_pos = ECEF2Local_NWU(ecef_pos, lat, long, alt, a, e2, 0)

                V_north = speed * math.cos(math.radians(bearing))
                V_west=-speed*math.sin(math.radians(bearing))
                before=nwu_pos
                nwu_pos=[nwu_pos[0]+V_north*dT,nwu_pos[1]+V_west*dT,nwu_pos[2]+V_up*dT]
                after=nwu_pos
                dx=after[0]-before[0]
                dy=after[1]-before[1]
                dist=math.sqrt(dx**2+dy**2)
                nwu_vel = [V_north, V_west, V_up]
                ecef_pos = Local_NWU2ECEF(nwu_pos, lat, long, alt, a, e2, 0)
                ecef_vel = Local_NWU2ECEF(nwu_vel, lat, long, alt, a, e2, 1)








                lat,long,alt=pm.ecef2geodetic(ecef_pos[0,0], ecef_pos[0,1], ecef_pos[0,2])


                #############################################
                Pos_Ref = ECEF2Local_NWU(ecef_pos, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2,
                                           0)
                Vel_Ref = ECEF2Local_NWU(ecef_vel, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2,
                                           1)

                # Update SV_Ref and CM_Ref in Updated_Track


########################################################
                # enu = list(pm.ecef2enu(ecef_pos[0], ecef_pos[1], ecef_pos[2], lat, long, alt))


                # nwu_vel = ECEF2Local_NWU(ecef_vel, lat, long, alt, a, e2, 1)
                # TBD \CONVERT VEL TO ECEF
                Dist_To_Dest=calc_distance(lat,long,next_lat,next_long)


                # rad
                a=1

                # print (bearing)
                # print(str(Dist_To_Dest),str(bearing))
                This_White_Picture = pd.DataFrame({
                    'ID': [sgid],
                    'Leg': [leg],
                    'Time': [time],
                    'Lat': [lat],
                    'Long': [long],
                    'Alt': [alt],
                    'Speed': [speed],
                    'Dist_To_Dest' :[Dist_To_Dest],
                    'bearing':[bearing],
                    'Pos_x':[ecef_pos[0,0]],
                    'Pos_y':[ecef_pos[0,1]],
                    'Pos_z':[ecef_pos[0,2]],
                    'Vel_x':[ecef_vel[0,0]],
                    'Vel_y':[ecef_vel[0,1]],
                    'Vel_z':[ecef_vel[0,2]],
                    'Pos_Ref' : [Pos_Ref],
                    'Vel_Ref' : [Vel_Ref]
                })
                a=1

                # np.array(White_Pos[1]).reshape(1, 3)


                White_Picture = pd.concat([White_Picture, This_White_Picture])
                # print(str(time), str(Dist_To_Dest),str(Prev_Dist_To_Dest))

                # print(Dist_To_Dest*1000,(Dist_To_Dest - Prev_Dist_To_Dest)*1000)
                delta1=Dist_To_Dest-Prev_Dist_To_Dest



                if Dist_To_Dest<0.1:
                    break
                Prev_Dist_To_Dest=Dist_To_Dest

                if speed<next_speed:
                    speed=speed+Hor_Acc*dT
                if speed>next_speed:
                    speed=speed-Hor_Acc*dT

                if alt > next_alt and V_up > 0:
                    V_up = -float(ThisWayPoints.VerticalSpeed[index])
                if alt < next_alt and V_up < 0:
                    V_up = float(ThisWayPoints.VerticalSpeed[index])

                # bearing = calc_bearing(lat, long, next_lat, next_long)

                bearing_to_next_leg = calc_bearing(lat, long, next_lat, next_long)
                Local_vel = ECEF2Local_NWU(ecef_vel, lat, long, alt, a, e2, 1)  # nwu

                dAlpha = Calc_dAlpha(math.radians(bearing_to_next_leg), math.radians(bearing))
                Radius = speed ** 2 / Turn_Acc

                # Circle circumference
                Full_Circle_Way = 2 * np.pi * Radius

                # Segment length
                Turn_Way = Full_Circle_Way * dAlpha / (2 * np.pi)

                # Duration (for constant speed t = s / v)
                Duration = abs(Turn_Way / speed)

                Average_Speed = (speed + next_speed) / 2
                # Nominal_Time = Dist_To_Dest / Average_Speed

                # Duration = min(Duration, Nominal_Time / 4)
                N = math.floor(Duration / dT)
                if N >= 1:
                    bearing = bearing + math.degrees(dAlpha) / N
                else:
                    bearing = bearing_to_next_leg
                time = time + dT
            a=1  # next time
        a=2 # next leg
    a=3 # next tagget
    return (White_Picture)

def ImportScenario(Traj,Params):

    import math
    import navpy
    import numpy as np
    from numpy import linalg as LA
    import pymap3d as pm
    from my_subscript import calc_bearing, calc_distance,getParam,ECEF2Local_NWU,Local_NWU2ECEF, Calc_dAlpha

    T_Start=0
    dT=getParam(Params, 'AT_Sys_Params.dT')
    a=getParam(Params, 'AT_Sys_Params.a')
    e2=getParam(Params, 'AT_Sys_Params.e2')

    Ref_Point_Lat = getParam(Params, 'AT_Sys_Params.Ref_Point_Lat')
    Ref_Point_Long = getParam(Params, 'AT_Sys_Params.Ref_Point_Long')
    Ref_Point_Alt = getParam(Params, 'AT_Sys_Params.Ref_Point_Alt')



    White_Picture=pd.DataFrame({
        'ID': [],
        'Time': [],
        'Leg': [],
        'Lat': [],
        'Long': [],
        'Alt': [],
        'Speed': [],
        'Dist_To_Dest': [],
        'bearing': [],
        'Pos_x':[],
        'Pos_y': [],
        'Pos_z': [],
        'Vel_x':[],
        'Vel_y': [],
        'Vel_z': [],
        'Pos_Ref': [],
        'Vel_Ref': []
    })
    White_Picture.Lat=Traj.lat
    White_Picture.Long = Traj.lon
    White_Picture.Alt = Traj.alt
    White_Picture.Pos_x = Traj.x_ecef
    White_Picture.Pos_y = Traj.y_ecef
    White_Picture.Pos_z = Traj.z_ecef

    White_Picture.Vel_x = Traj.vx_ecef
    White_Picture.Vel_y = Traj.vy_ecef
    White_Picture.Vel_z = Traj.vz_ecef

    White_Picture.Time = Traj.Time
    White_Picture.Speed = Traj.Speed
    White_Picture.bearing=Traj.heading

    White_Picture.ID=1

    return (White_Picture)



    pass


def Visualization(ScenarioFolder, White_Picture, Drone_Plots, Sensors, Drone_Tracks):
    # ScenarioFolder = "C:\Drones\Python\Scenarios\\" + Scenario + "\\"
    import pickle
    from folium.plugins import MeasureControl

    try:
        try:
            IDs = White_Picture['ID'].unique().tolist()
        except:
            sgid_values = [int(item.item()) for item in White_Picture['sgid']]
            IDs = list(set(sgid_values))  # Get unique values

        fig = plt.figure(num='LLA')


        for sgid in IDs:
            # This_White_Picture = White_Picture[White_Picture['SimID'] == sgid].copy()
            # ThisWayPoints = WayPoints[WayPoints['ID'] == sgid].copy()
            ThisWhitePicture = White_Picture[White_Picture['ID'] == sgid].copy()
            ThisPlots = Drone_Plots[Drone_Plots['SimId'] == sgid].copy()
            ThisTracks = Drone_Tracks[Drone_Tracks['SimId'] == sgid].copy()
            ThisTracks = ThisTracks[ThisTracks['Alive'] == 1].copy()

            plt.plot(ThisWhitePicture.Long, ThisWhitePicture.Lat, '->')

            plt.plot(ThisPlots.Long, ThisPlots.Lat, '+')

            plt.plot(ThisTracks.Long, ThisTracks.Lat, '-v')
        plt.plot(Sensors.Long, Sensors.Lat, '*')

        plt.grid()
        FileName = "LLA.png"
        fig.savefig(ScenarioFolder + FileName)
        print('LLA complete')



        with open(ScenarioFolder + "LLA.pkl", "wb") as f:  # Open file in write-binary mode
            pickle.dump(fig, f)  # Dump the figure into the file

        # Access value













        if True:


            ID_values = Drone_Tracks['ID'].unique()
            for ID in ID_values:

                This_Track = Drone_Tracks[Drone_Tracks['ID'] == ID].copy()
                fig = plt.figure(num='Track '+ str(int(ID))+ ' Target '+str(This_Track['SimId'].iloc[0])+' Kinematics')



                axes = np.array([
                    fig.add_subplot(2, 2, 1),
                    fig.add_subplot(2, 2, 2),
                    fig.add_subplot(2, 2, 3),
                    fig.add_subplot(2, 2, 4)
                ]).reshape(2, 2)



                fig.suptitle('Track ' + str(int(ID)) + ' Target ' + str(This_Track['SimId'].iloc[0]))
                Pos_array = np.array(This_Track['Pos'].tolist())  # Shape: (n, 3)
                Vel_array = np.array(This_Track['Vel'].tolist())  # Shape: (n, 3)

                Time = This_Track['Time']

                ax = axes[0, 0]

                ax.plot(Time, Pos_array[:, 0], label='X')
                ax.plot(Time, Pos_array[:, 1], label='Y')
                ax.plot(Time, Pos_array[:, 2], label='Z')

                ax.set_xlabel('Time [sec')
                ax.set_ylabel('Position [m]')
                ax.legend()

                ax.grid(True)

                ax = axes[0, 1]

                ax.plot(Time, Vel_array[:, 0], label='X')
                ax.plot(Time, Vel_array[:, 1], label='Y')
                ax.plot(Time, Vel_array[:, 2], label='Z')

                ax.set_xlabel('Time [sec')
                ax.set_ylabel('Velocity [m/sec]')
                ax.legend()

                ax.grid(True)

                ax = axes[1, 0]

                ax.plot(Time, This_Track['Heading'], label='Heading')

                ax.set_xlabel('Time [sec')
                ax.set_ylabel('Heading [deg]')
                ax.legend()

                ax.grid(True)

                ax = axes[1, 1]

                ax.plot(Time, This_Track['Speed'], label='Speed')

                speed = This_Track['Speed']
                mean_speed = np.mean(speed)
                std_speed = np.std(speed)
                ax.set_title(f"Speed vs Time Mean: {mean_speed:.2f}, Std: {std_speed:.2f}")

                ax.set_xlabel('Time [sec')
                ax.set_ylabel('Speed [m/sec]')

                ax.grid(
                    True)

                FileName = "Track" + str(int(ID)) + ".png"
                fig.savefig(ScenarioFolder + FileName)



            print('Folium start')

            import folium
            import pandas as pd
            import webbrowser
            from folium.plugins import MousePosition


            # Example DataFrame (Replace with your actual data)
            Updated_Tracks = Drone_Tracks[Drone_Tracks['Extrapolated'] == 0].copy()
            Extrapolated_Tracks = Drone_Tracks[Drone_Tracks['Extrapolated'] == 1].copy()
            Updated_Tracks = Drone_Tracks[Drone_Tracks['Extrapolated'] == 0].copy()
            Radars = Sensors[Sensors['SensorType'] == 1]
            Sigint_Sensors = Sensors[Sensors['SensorType'] == 2]


            # Create the map centered at the first coordinate
            m = folium.Map(location=[White_Picture["Lat"].iloc[0], White_Picture["Long"].iloc[0]], zoom_start=13,tiles = None)




            m.add_child(MeasureControl(primary_length_unit='meters'))
            MousePosition().add_to(m)

            # Coordinates for Google Street View (example: New York)
            # street_view_url = "https://maps.googleapis.com/maps/api/streetview?size=600x300&location=51.5,-0.5 0&key=YOUR_GOOGLE_API_KEY"

            # Add an iframe with the Street View image
            # iframe = folium.IFrame(f'<img src="{street_view_url}" width="600" height="300">', width=600, height=300)
            # popup = folium.Popup(iframe, max_width=600)

            # Feature groups for legend
            fg_blue = folium.FeatureGroup(name="True Path (Blue)").add_to(m)
            fg_red = folium.FeatureGroup(name="Plots (Red)").add_to(m)
            fg_green = folium.FeatureGroup(name="Updated Tracks (Green)").add_to(m)
            fg_yellow = folium.FeatureGroup(name="Radar (Yellow)").add_to(m)
            fg_cyan = folium.FeatureGroup(name="Sigint (Cyan)").add_to(m)
            fg_black = folium.FeatureGroup(name="Extrapolated Tracks (Black)").add_to(m)

            # Add first path (Blue) – full path remains
            Lat_col_index = White_Picture.columns.get_loc('Lat')
            Long_col_index = White_Picture.columns.get_loc('Long')

            lat_col = White_Picture.iloc[:, Lat_col_index]
            long_col = White_Picture.iloc[:, Long_col_index]

            folium.PolyLine(
                list(zip(lat_col, long_col)),
                color="blue", weight=3, opacity=0.7
            ).add_to(fg_blue)
            print('Folium WP complete')

            # Filter out NaN values for Lat2 and Long2 **without affecting Lat1**


            Lat_col_index = Drone_Plots.columns.get_loc('Lat')
            Long_col_index =Drone_Plots.columns.get_loc('Long')

            lat_col =Drone_Plots.iloc[:, Lat_col_index]
            long_col =Drone_Plots.iloc[:, Long_col_index]


            # Add red markers (not connected)
            for lat, lon in zip(lat_col, long_col):
                if not math.isnan(lat):
                    folium.RegularPolygonMarker(
                        location=[lat, lon],
                        radius=2,  # Marker size
                        color="red",
                        fill=True,
                        fill_color="red",
                        fill_opacity=0.8
                    ).add_to(fg_red)
            print('Folium Plots complete')

            Lat_col_index = Updated_Tracks.columns.get_loc('Lat')
            Long_col_index =Updated_Tracks.columns.get_loc('Long')

            lat_col =  Updated_Tracks.iloc[:, Lat_col_index]
            long_col =  Updated_Tracks.iloc[:, Long_col_index]

            for lat, lon in zip(lat_col, long_col):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,  # Marker size
                    color="green",
                    fill=True,
                    fill_color="green",
                    fill_opacity=0.8
                ).add_to(fg_green)
            print('Folium Tracks complete')



            Lat_col_index = Radars.columns.get_loc('Lat')
            Long_col_index = Radars.columns.get_loc('Long')

            lat_col =  Radars.iloc[:, Lat_col_index]
            long_col =  Radars.iloc[:, Long_col_index]

            for lat, lon in zip(lat_col, long_col):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,  # Marker size
                    color="yellow",
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=0.8
                ).add_to(fg_yellow)

                Lat_col_index = Sigint_Sensors.columns.get_loc('Lat')
                Long_col_index = Sigint_Sensors.columns.get_loc('Long')

                lat_col = Sigint_Sensors.iloc[:, Lat_col_index]
                long_col = Sigint_Sensors.iloc[:, Long_col_index]

                for lat, lon in zip(lat_col, long_col):
                    if not math.isnan(lat):


                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=6,  # Marker size
                            color="cyan",
                            fill=False,
                            fill_color="cyan",
                            fill_opacity=0.8
                        ).add_to(fg_cyan)

            print('Folium Sensors complete')

            Lat_col_index = Extrapolated_Tracks.columns.get_loc('Lat')
            Long_col_index = Extrapolated_Tracks.columns.get_loc('Long')

            lat_col = Extrapolated_Tracks.iloc[:, Lat_col_index]
            long_col = Extrapolated_Tracks.iloc[:, Long_col_index]

            for lat, lon in zip(lat_col, long_col):


                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,  # Marker size
                    color="black",
                    fill=True,
                    fill_color="black",
                    fill_opacity=0.8
                ).add_to(fg_black)
            print('Folium Extrap complete')

            # Add layer control (legend)
            Sigint_Sensors=Sensors[Sensors['SensorType']==2]
            Sigint_Sensors_IDs=Sigint_Sensors['SensorId'].unique()

            # Create a FeatureGroup for the lines
            lines_layer = folium.FeatureGroup(name='Direction Lines')

            for ID in Sigint_Sensors_IDs:
                This_Sigint_Sensor=Sigint_Sensors[Sigint_Sensors['SensorId']==ID]
                This_Sigint_Detections=Drone_Plots[Drone_Plots['SensorId']==ID]
                Filtered_Sigint_Detections = This_Sigint_Detections.iloc[::5]
                # Inputs
                start_lat = This_Sigint_Sensor['Lat'].iloc[0]  # example: Tel Aviv
                start_lon = This_Sigint_Sensor['Long'].iloc[0]
                for Det in Filtered_Sigint_Detections.itertuples(index=False):
                    azimuth_deg = math.degrees(Det.Az)  # degrees, clockwise from north
                    length_km = 5  # how far the line should go

                    # Calculate destination point using azimuth
                    def destination_point(lat, lon, azimuth_deg, distance_km):
                        origin = (lat, lon)
                        # geopy expects azimuth in degrees clockwise from North, just like you have
                        return geodesic(kilometers=distance_km).destination(origin, azimuth_deg)

                    end_point = destination_point(start_lat, start_lon, azimuth_deg, length_km)

                    # Create folium map centered at start point
                    # m = folium.Map(location=[start_lat, start_lon], zoom_start=13)

                    # Draw line
                    folium.PolyLine(locations=[
                        (start_lat, start_lon),
                        (end_point.latitude, end_point.longitude)
                    ], color='black', weight=2).add_to(lines_layer)
            lines_layer.add_to(m)

                    # Optional: mark the start and end
                    # folium.Marker(location=[start_lat, start_lon], popup='Start').add_to(m)
                    # folium.Marker(location=[end_point.latitude, end_point.longitude], popup='End').add_to(m)


            print('Folium Sigint complete')



            # Custom legend using HTML and JavaScript
            legend_html = """
            <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 150px; height: 90px; 
             background-color: white; z-index:9999; 
             font-size:10px; border:2px solid grey;
             padding: 10px;">
             <b>Legend</b><br>
                <svg width="10" height="10"><rect width="10" height="10" style="fill:blue;stroke-width:1;stroke:black"/></svg> True Path (Blue)<br>
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" style="fill:red;stroke:red"/></svg> Plots (Red) <br>
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" style="fill:black;stroke:green"/></svg> Tracks (Green) <br>
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" style="fill:yellow;stroke:yellow"/></svg> Radar (Yellow) <br>
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" style="fill:yellow;stroke:yellow"/></svg> Sigint (Cyan) <br>
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" style="fill:black;stroke:black"/></svg> Extrap (Black) <br>
            </div>
            """

            m.get_root().html.add_child(folium.Element(legend_html))
            # folium.TileLayer('Esri.WorldImagery').add_to(m)



            # Save and open the map




            folium.TileLayer(
                'CartoDB positron',
                attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
            ).add_to(m)

            folium.TileLayer(
                'CartoDB dark_matter',
                attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
            ).add_to(m)

            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(m)

            folium.TileLayer(
                tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                attr='© OpenTopoMap contributors',
                name='Topo Map'
            ).add_to(m)

            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles © Esri',
                name='Esri Satellite'
            ).add_to(m)

            folium.TileLayer(
                tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                attr='© OpenStreetMap contributors © CARTO',
                name='Carto Light'
            ).add_to(m)

            folium.TileLayer('OpenStreetMap').add_to(m)

            folium.LayerControl().add_to(m)

            map_file = "index.html"
            m.save(ScenarioFolder+map_file)
            webbrowser.open(ScenarioFolder+map_file)

        print("Map saved as map.html. Open it in a browser.")

        print('showing plots')
        plt.show(block=False)

        plt.pause(0.5)
        print('thank you')
        a = 1
    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        a = 1



def Show_Tracks_Errors(ScenarioFolder, Drone_Tracks, Sensors, White_Picture):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    IDs = np.unique(Drone_Tracks['ID'])

    for k in range(len(IDs)):
        This_Track = Drone_Tracks[(Drone_Tracks['ID'] == IDs[k]) & (Drone_Tracks['Extrapolated'] == 0)]

        _, ind = np.unique(This_Track['Time'], return_index=True)
        # ind = ind[ind > 4]
        This_Track = This_Track.iloc[ind]

        Show_Single_Track_Errors(ScenarioFolder, This_Track, Sensors, White_Picture,IDs[k])


def Show_Single_Track_Errors(ScenarioFolder, This_Track, Sensors, White_Picture,ID):
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import traceback
    from scipy.stats import mode, cumfreq, chi2
    from my_subscript import my_cdfplot

    try:

        dSV_Tracks = []
        Chi2 = []
        Chi2_Pos = []
        Chi2_Vel = []

        # Extract and flatten
        SimId = np.array(This_Track["SimId"]).flatten()

        # Filter out non-integer values (e.g., None or strings)
        SimId = np.array([x for x in SimId if isinstance(x, int)], dtype=int)

        # Find the most frequent value

        SimId = np.bincount(SimId).argmax()
        This_White_Picture = White_Picture[White_Picture['ID'] == SimId]

        dSV = []
        dPos = []
        dVel = []
        Sigma = []
        SigmaPos = []
        SigmaVel = []

        #  From python convert series to array of lists
        # White_Pos = np.array(This_White_Picture["Pos_Ref"])
        # White_Pos = np.array([np.array(item).reshape(1, 3) for item in White_Pos])

        # White_Pos = np.array([np.array(item).reshape(1, 3) for item in White_Pos])
        White_Pos = np.array(This_White_Picture["Pos_Ref"])
        White_Pos = np.array(White_Pos.tolist(), dtype=np.float64)

        if White_Pos.ndim == 1:  # matlab
            White_Pos = White_Pos.reshape(-1, 1, 3)

        White_Vel = np.array(This_White_Picture["Vel_Ref"])
        White_Vel = np.array(White_Vel.tolist(), dtype=np.float64)
        if White_Vel.ndim == 1:  # matlab
            White_Vel = White_Vel.reshape(-1, 1, 3)

        White_Time = np.array(This_White_Picture["Time"])

        Pos_interp_func = interp1d(White_Time, White_Pos, axis=0, kind='linear')
        Vel_interp_func = interp1d(White_Time, White_Vel, axis=0, kind='linear')

        for j in range(len(This_Track['Time'])):

            CM = np.array(This_Track['CM_Ref'].iloc[j])[0]
            Pos = np.array(This_Track['Pos_Ref'].iloc[j][0])
            Vel = np.array(This_Track['Vel_Ref'].iloc[j][0])
            SigmaPos.append(math.sqrt(np.trace((CM[0:3, 0:3]))))
            SigmaVel.append(math.sqrt(np.trace((CM[3:6, 3:6]))))

            time = This_Track['Time'].iloc[j]

            try:

                # Get interpolated white position and velocity
                Pos_White = Pos_interp_func(time)
                # the alternative is to implement directly without creating it as follows
                Pos_White1 = interp1d(White_Time, White_Pos, axis=0, kind='linear')(time)
                Vel_White = Vel_interp_func(time)

                dPos0 = Pos - Pos_White
                dVel0 = Vel - Vel_White
                dSV0 = np.concatenate((dPos0, dVel0 ))

                # [np.vstack((dPos0.reshape(-1, 1), dVel0.reshape(-1, 1)]
                dPos.append(dPos0)
                dVel.append(dVel0)
                dSV.append(dSV0)

                # dSV.append (np.vstack((dPos0.reshape(-1, 1), dVel0.reshape(-1, 1))))  # Now (6,1)

                Chi2.append(dSV0 @ np.linalg.inv(CM) @ dSV0.T)
                Chi2_Pos.append(dPos0 @ np.linalg.inv(CM[0:3, 0:3]) @ dPos0.T)
                Chi2_Vel.append(dVel0 @ np.linalg.inv(CM[3:6, 3:6]) @ dVel0.T)
                a = 1

                # dSV.append(This_Track['SV_Ref'].iloc[j, :6] - interp_func(This_Track['Time'].iloc[j]))
            except Exception as err:
                print(err)
        Chi2_Sorted = np.sort(np.array(Chi2))
        Chi2_Stat = np.linspace(np.min(Chi2_Sorted), np.max(Chi2_Sorted), 100)

        Pos_Err = np.linalg.norm(dPos, axis=1)
        Vel_Err = np.linalg.norm(dVel, axis=1)

        # plt.ion()  # Turn on interactive mode

        for cnt in range(3):
            if cnt==0:
                Part=' Pos Err'
            elif cnt==1:
                Part=' Vel Err'
            elif cnt==2:
                Part=' Total Err'



            fig = plt.figure(num='Track '+str(int(This_Track['ID'].iloc[0]))+ ' Target ' + str(int(This_Track['SimId'].iloc[0]))+ Part, figsize=(10, 8))




            if cnt<=1:
                gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 columns

                ax1 = fig.add_subplot(gs[0, 0])  # Row 0, Col 0
                ax2 = fig.add_subplot(gs[0, 1])  # Row 0, Col 1
                ax3 = fig.add_subplot(gs[1, :])  # Row 1, span both columns

                # Example content:
                ax1.set_title(" Errors")
                ax2.set_title(r"$\chi^2$ vs Time")
                ax3.set_title(r"$\chi^2$ cdf")


            else:
                gs = gridspec.GridSpec(1, 2)  # 1 rows, 2 columns

                ax2 = fig.add_subplot(gs[0, 0])  # Row 0, Col 0
                ax3 = fig.add_subplot(gs[0, 1])  # Row 0, Col 1



                ax2.set_title(r"$\chi^2$ vs Time")
                ax3.set_title(r"$\chi^2$ cdf")


                # Example content:









            ax1.grid()
            ax1.set_xlabel("Time")

            if cnt == 0:

                ax1.set_ylabel("Pos Err [m]")
                ax1.plot(This_Track['Time'], Pos_Err, '-bo', label="True Pos Err")
                ax1.plot(This_Track['Time'], -Pos_Err, '-bo', label="-True Pos Err")

                ax1.plot(This_Track['Time'], np.array(SigmaPos), '-r', label="Est Pos Err")
                ax1.plot(This_Track['Time'], -np.array(SigmaPos), '-r', label="-Est Pos Err")


            elif cnt==1:
                ax1.set_ylabel("Vel Err [m/s]")
                ax1.plot(This_Track['Time'], Vel_Err, '-bo', label="True Vel Err")
                ax1.plot(This_Track['Time'], -Vel_Err, '-bo', label="-True Vel Err")

                ax1.plot(This_Track['Time'], np.array(SigmaVel), '-r', label="Est Vel Err")
                ax1.plot(This_Track['Time'], -np.array(SigmaVel), '-r', label="-Est Vel Err")

            ax1.legend()


            ax2.grid()
            ax2.set_xlabel("Time")

            if cnt == 0:
                ax2.set_ylabel(r'$\chi^2$')
                ax2.plot(np.array(This_Track['Time']), np.array(Chi2_Pos).squeeze(), '-bo',
                        label=r'Pos  Err $\chi^2$')


            elif cnt==1:
                ax2.set_ylabel(r'$\chi^2$')
                ax2.plot(This_Track['Time'], np.array(Chi2_Vel).squeeze(), '-bo', label=r'Vel  Err $\chi^2$')
            elif cnt == 2:
                ax2.set_ylabel(r'$\chi^2$')
                ax2.plot(This_Track['Time'], np.array(Chi2).squeeze(), '-bo', label=r'Vel  Err $\chi^2$')

            ax2.legend()

            # Track_ID = This_Track['ID'].iloc[0]
            # Save_Name = f"{Folder}/{'Matlab' if mm == 1 else 'C'} Tracks_{'Pos' if cnt == 1 else 'Vel'}{Track_ID}.png"
            # plt.savefig(Save_Name)


            ax3.grid()

            # Generate some sample data (for example, random normal distribution)
            if cnt <= 1:
                dim=3
            else:
                dim=6


            chi2_samples = np.random.chisquare(dim, size=1000)


            x = np.linspace(0, np.max(chi2_samples), 1000)  # Range of values
            cdf_values0 = chi2.cdf(x, dim)  # Compute CDF

            # Plot CDF
            if cnt <= 1:
                ax3.plot(x, cdf_values0, '-o', color='blue', label=r'CDF of $\chi^2$ (df=3)')
            else:
                ax3.plot(x, cdf_values0, '-o', color='blue', label=r'CDF of $\chi^2$ (df=6)')
            if cnt == 0:
                Chi2_Pos_Sorted = np.sort(np.array(Chi2_Pos))  # Ensure sorted for proper CDF
                # hi2_Pos_Stat = np.linspace(np.min(Chi2_Pos_Sorted), np.max(Chi2_Pos_Sorted), 100)
                # cdf_values = chi2.cdf(Chi2_Pos_Stat, 3)
                # ax.plot(Chi2_Pos_Stat, cdf_values, '-x',color='black', label=r'Pos errors $\chi^2$')
                # Compute CDF
                XData, YData = my_cdfplot(Chi2_Pos_Sorted, 100)
                ax3.plot(XData, YData, '-x', color='black', label=r'Pos errors1 $\chi^2$')
                FileName = "PosErrors"+str(int(ID))+".png"


                fig.savefig(ScenarioFolder + FileName)



            elif cnt==1:
                Chi2_Vel_Sorted = np.sort(np.array(Chi2_Vel))  # Ensure sorted for proper CDF
                # Chi2_Vel_Stat = np.linspace(np.min(Chi2_Vel_Sorted), np.max(Chi2_Vel_Sorted), 100)
                # cdf_values = chi2.cdf(Chi2_Vel_Stat, 3)  # Compute CDF
                # ax.plot(Chi2_Vel_Stat, cdf_values, '-x',color='black', label=r'Vel errors $\chi ^ 2$')

                XData, YData = my_cdfplot(Chi2_Vel_Sorted, 100)
                ax3.plot(XData, YData, '-x', color='black', label=r'Vel errors1 $\chi^2$')


                FileName = "VelErrors" + str(int(ID)) + ".png"
                plt.suptitle('Track '+str(int(This_Track['ID'].iloc[0]))+ ' Target '+str(int(This_Track['SimId'].iloc[0])) , fontsize=16)
                fig.savefig(ScenarioFolder + FileName)
            elif cnt==2:
                Chi2_Sorted = np.sort(np.array(Chi2))  # Ensure sorted for proper CDF
            # Chi2_Vel_Stat = np.linspace(np.min(Chi2_Vel_Sorted), np.max(Chi2_Vel_Sorted), 100)
            # cdf_values = chi2.cdf(Chi2_Vel_Stat, 3)  # Compute CDF
            # ax.plot(Chi2_Vel_Stat, cdf_values, '-x',color='black', label=r'Vel errors $\chi ^ 2$')

                XData, YData = my_cdfplot(Chi2_Sorted, 100)
                ax3.plot(XData, YData, '-x', color='black', label=r'Total errors1 $\chi^2$')

                FileName = "TotalErrors" + str(int(ID)) + ".png"

                fig.savefig(ScenarioFolder + FileName)
            plt.suptitle(
                'Track ' + str(int(This_Track['ID'].iloc[0])) + ' Target ' + str(int(This_Track['SimId'].iloc[0])),
                fontsize=16)


            ax3.set_xlabel('Value')
            ax3.set_ylabel('Cumulative Probability')
            ax3.legend()

        # Plot the CDF


            plt.show(block=False)

            plt.pause(0.5)






    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        a = 1


def Show_Plots_Errors(ScenarioFolder, Drone_Plots, Sensors, White_Picture, Params):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    IDs = np.unique(Drone_Plots['SimId'])

    for k in range(len(IDs)):
        This_White_Picture = White_Picture[White_Picture['ID'] == IDs[k]]
        This_Target_Plot = Drone_Plots[Drone_Plots['SimId'] == IDs[k]]
        SIDs = np.unique(This_Target_Plot['SensorId'])
        for m in range(len(SIDs)):
            This_Plot = This_Target_Plot[This_Target_Plot['SensorId'] == SIDs[m]]

            # _, ind = np.unique(This_Plot['Time'], return_index=True)
            # This_Plot = This_Plot.iloc[ind]
            This_Radar = Sensors[Sensors['SensorId'] == SIDs[m]]

            Show_Single_Plot_Errors(ScenarioFolder, This_Plot, This_Radar, This_White_Picture, Params)


def Show_Single_Plot_Errors(ScenarioFolder, This_Plot, This_Radar, This_White_Picture, Params):
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import traceback
    from scipy.stats import mode, cumfreq, chi2
    from my_subscript import my_cdfplot, getParam, ECEF2Local_NWU, enu2spherical_rdot, nwu2enu, Calc_dAlpha

    a = getParam(Params, 'AT_Sys_Params.a')
    e2 = getParam(Params, 'AT_Sys_Params.e2')

    try:
        # This_Radar = Sensors[Sensors['SensorId'] == This_Plot["SensorId"].iloc[0]]

        SimId = This_Plot.iloc[0]['SimId']
        Radar_Id = This_Radar['SensorId'].iloc[0]
        Label = 'Target ' + str(int(SimId)) + ' Radar ' + str(Radar_Id)

        # This_White_Picture = White_Picture[White_Picture['ID'] == SimId]

        #  From python convert series to array of lists
        White_Pos = np.array([np.array(This_White_Picture["Pos_x"]), np.array(This_White_Picture["Pos_y"]),
                              np.array(This_White_Picture["Pos_z"])]).T
        White_Vel = np.array([np.array(This_White_Picture["Vel_x"]), np.array(This_White_Picture["Vel_y"]),
                              np.array(This_White_Picture["Vel_z"])]).T

        White_Time = np.array(This_White_Picture["Time"])

        Pos_interp_func = (
            interp1d(White_Time, White_Pos, axis=0, kind='linear'))
        Vel_interp_func = interp1d(White_Time, White_Vel, axis=0, kind='linear')

        try:

            # Get interpolated white position and velocity
            Chi2 = []
            MeasError = np.empty((0, 4))
            for index, row in This_Plot.iterrows():
                This_Plot1 = row
                time = This_Plot1.Time
                Pos_White = Pos_interp_func(time)  # ecef
                # the alternative is to implement directly without creating it as follows

                Vel_White = Vel_interp_func(time)

                NWU_Pos = ECEF2Local_NWU(Pos_White.reshape(3, 1), This_Radar.Lat, This_Radar.Long,
                                         This_Radar.Height, a, e2, 0)  # ecef
                NWU_Vel = ECEF2Local_NWU(Vel_White.reshape(3, 1), This_Radar.Lat, This_Radar.Long,
                                         This_Radar.Height, a, e2, 1)  # ecef

                SV_Sphere = enu2spherical_rdot(nwu2enu(np.array(NWU_Pos)), nwu2enu(np.array(NWU_Vel)))
                dSV = np.array([
                    This_Plot1.Range - SV_Sphere[0],
                    Calc_dAlpha(This_Plot1.Az, SV_Sphere[1]),
                    Calc_dAlpha(This_Plot1.El, SV_Sphere[2]),
                    This_Plot1.Doppler - SV_Sphere[3]])

                CM_Sphere_Plot = np.diag([
                    This_Plot1.RangeError ** 2,
                    This_Plot1.AzError ** 2,
                    This_Plot1.ElError ** 2,
                    This_Plot1.DopplerError ** 2
                ])

                Chi2.append(dSV @ np.linalg.inv(CM_Sphere_Plot) @ dSV.T)
                MeasError = np.vstack([MeasError, dSV])

                a = 1

            # dSV.append(This_Plot['SV_Ref'].iloc[j, :6] - interp_func(This_Plot['Time'].iloc[j]))
        except Exception as err:
            print(err)
        Chi2_Sorted = np.sort(np.array(Chi2))
        Chi2_Stat = np.linspace(np.min(Chi2_Sorted), np.max(Chi2_Sorted), 100)

        # plt.ion()  # Turn on interactive mode

        fig = plt.figure(figsize=(100, 8), constrained_layout=True,num='Plot for Radar '+str(int(This_Radar['SensorId']))+' Target '+str(int(This_Plot['SimId'].iloc[0])))
        gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1])

        # Row 1: 4 plots in columns 0–3
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])

        # Row 2: 2 wide plots
        ax5 = fig.add_subplot(gs[1, 0:2])  # spans columns 0 and 1
        ax6 = fig.add_subplot(gs[1, 2:4])  # spans columns 2 and 3

        ax1.set_xlabel("Time")

        ax1.set_ylabel('Range Err')
        ax1.plot(np.array(This_Plot['Time']), MeasError[:,0], '-bo', label=r' dRange')
        ax1.grid()

        ax2.set_xlabel("Time")

        ax2.set_ylabel('Az Err')
        ax2.plot(np.array(This_Plot['Time']), MeasError[:, 1], '-bo', label=r' dAz')
        ax2.grid()


        ax3.set_xlabel("Time")

        ax3.set_ylabel('El Err')
        ax3.plot(np.array(This_Plot['Time']), MeasError[:, 2], '-bo', label=r' dEl')
        ax3.grid()

        ax4.set_xlabel("Time")

        ax4.set_ylabel('Dop Error')
        ax4.plot(np.array(This_Plot['Time']), MeasError[:, 3], '-bo', label=r' dDop')
        ax4.grid()




        ax5.set_xlabel("Time")

        ax5.set_ylabel(r'$\chi^2$')
        ax5.plot(np.array(This_Plot['Time']), np.array(Chi2).squeeze(), '-bo', label=r' $\chi^2$')
        ax5.grid()





        # Generate some sample data (for example, random normal distribution)
        chi2_samples = np.random.chisquare(4, size=10000)
        x = np.linspace(0, np.max(chi2_samples), 10000)  # Range of values
        cdf_values0 = chi2.cdf(x, 4)  # Compute CDF

        # Plot CDF
        ax6.plot(x, cdf_values0, '-o', color='blue', label=r'CDF of $\chi^2$ distribution (df=4)')



        Chi2_Sorted = np.sort(np.array(Chi2))  # Ensure sorted for proper CDF

        XData, YData = my_cdfplot(Chi2_Sorted, 100)
        ax6.plot(XData, YData, '-x', color='black', label=r'True $\chi^2$ CDF')
        ax6.grid()
        plt.suptitle(Label, fontsize=16)
        FileName = Label
        fig.savefig(ScenarioFolder + FileName)



        ax6.set_xlabel('Value')
        ax6.set_ylabel('Cumulative Probability')
        ax6.legend()



        print('before show')
        plt.show(block=False)

        plt.pause(0.5)
        print('after show')
        a = 2





    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
