import warnings
import sys
from my_subscript import BuildParams, BuildEPL
import pandas as pd
import math

import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm
from my_subscript import BuildParams
import warnings
import sys
import traceback

def BuildDairPlots(White_Picture,Sensors,Params,EPL):
    import pandas as pd
    try:
        print("Pandas version:", pd.__version__)
        pd.options.mode.chained_assignment = 'raise'  # Raises SettingWithCopyError
    except NameError as e:
        print("Error:", e)
        print("Pandas is not imported correctly.")









    pd.options.mode.chained_assignment = 'raise'  # Raises SettingWithCopyError


    try:
        print ('Building Plots ')
        from my_subscript import calc_bearing, calc_distance, getParam,ECEF2Local_NWU, Local_NWU2ECEF,nwu2enu,enu2spherical_rdot,Calc_dAlpha,Shere2Cart
        a = getParam(Params, 'AT_Sys_Params.a')
        e2 = getParam(Params, 'AT_Sys_Params.e2')
        BearingRateError=getParam(Params, 'AT_Sys_Params.BearingRateError')
        import numpy as np
        import pandas as pd
      #   from Radar_Functions import snr_estimation_DAIR,Pd_case1,Pd_case3




        # dT = getParam(Params, 'AT_Sys_Params.dT')
        Window_Width=getParam(Params,'AT_Sys_Params.Window_Width')
        try:
            IDs = White_Picture['ID'].unique().tolist()
        except:
            sgid_values = [int(item.item()) for item in White_Picture['sgid']]
            IDs = list(set(sgid_values))  # Get unique values

        Drone_Plots = pd.DataFrame({
            'Time': [],
            'SimId': [],
            'SensorId': [],
            'SensorType': [],
            'RangeError': [],
            'AzError': [],
            'ElError': [],
            'DopplerError': [],
            'Range': [],
            'Az': [],
            'El': [],
            'Doppler': [],
            'x': [],
            'y': [],
            'z': [],
            'CM_Cart': [],
            'Lat': [],
            'Long': [],
            'Alt': [],
            'ECEF_X':[],
            'ECEF_Y':[],
            'ECEF_Z':[],
            'Az_Rate': [],
            'Az_Rate_Error':[],
            'CM_Polar':[],
            'SPAN_High':[],
            'SPAN_Low':[],
            'Remote_Control':[],
            'id':[],
            'EPL_Name':[],
            'Frequency_Low':[],
            'Frequency_High':[],
            'PRI_Low':[],
            'PRI_High':[],
            'PW1_Low':[],
            'PW1_High':[],
            'Group_ID':[],

        })
        for sgid in IDs:
            # This_White_Picture = White_Picture[White_Picture['ID'] == sgid].copy()
            # ThisWayPoints = WayPoints[WayPoints['ID'] == sgid].copy()
            try:
                ThisWhitePicture = White_Picture[White_Picture['ID'] == sgid].copy()
            except:
                ThisWhitePicture = White_Picture[White_Picture['sgid'] == sgid].copy()
            This_EPL=EPL[EPL['id']==sgid].copy()


            Scenario_Start = min(ThisWhitePicture.Time)
            Scenario_End = max(ThisWhitePicture.Time)

            Times=np.arange(Scenario_Start,Scenario_End,Window_Width)


            # tbd handle multi targets
            for index,row in Sensors.iterrows():
                print ('Processing Target '+str(sgid)+ 'Sensor '+str(index))
                Trot = row.RotationTime
                scanrate=1/Trot
                Tdwell=getParam(Params,'AT_Sys_Params.HPBW')/scanrate
                prf=getParam(Params,'AT_Sys_Params.PRF_hz')
                pi=1/prf
                az0_deg = np.random.uniform(low = 0.0, high = 1.0, size = None)   * 360;
                # az0_deg=0
                Sensor_Lat=row.Lat
                Sensor_Long=row.Long
                Sensor_Height=row.Height
                MinElevation=row.MinElevation
                MaxElevation = row.MaxElevation
                MinRange = row.MinRange
                MaxRange = row.MaxRange
                AzWidth= row.AzimuthBW
                This_Sensor=row


                Iterations_Scan = (1 /Trot) / getParam(Params,'AT_Sys_Params.Window_Width')





                Last_Plot = pd.DataFrame()
                for idx, time in enumerate(Times):
                    prob_cond=0
                    Center_Az1=(az0_deg+time*Trot)
                    Center_Az=Center_Az1 % 360

                    # print (time,Center_Az)
                    # index_Times=np.array(index.Time)
                    # t=pd.to_numeric(index.Time)

                    t=ThisWhitePicture.Time

                    try:
                        x =pd.to_numeric(ThisWhitePicture.Pos_x)
                        y = pd.to_numeric(ThisWhitePicture.Pos_y)
                        z = pd.to_numeric(ThisWhitePicture.Pos_z)

                        vx = pd.to_numeric(ThisWhitePicture.Vel_x)
                        vy = pd.to_numeric(ThisWhitePicture.Vel_y)
                        vz = pd.to_numeric(ThisWhitePicture.Vel_z)

                        Pos_x = np.interp(time, t, x)
                        Pos_y = np.interp(time, t, y)
                        Pos_z = np.interp(time, t, z)

                        Vel_x = np.interp(time, t, vx)
                        Vel_y = np.interp(time, t, vy)
                        Vel_z = np.interp(time, t, vz)

                    except:  # matlab origin
                        x = ThisWhitePicture.Pos_x
                        y = ThisWhitePicture.Pos_y
                        z = ThisWhitePicture.Pos_z

                        vx = ThisWhitePicture.Vel_x
                        vy = ThisWhitePicture.Vel_y
                        vz = ThisWhitePicture.Vel_z

                        t_values = [val[0][0] for val in t]
                        x_values = [val[0][0] for val in x]  # No need to unpack the index part
                        y_values = [val[0][0] for val in y]
                        z_values = [val[0][0] for val in z]

                        vx_values = [val[0][0] for val in vx]  # No need to unpack the index part
                        vy_values = [val[0][0] for val in vy]
                        vz_values = [val[0][0] for val in vz]

                        # Now use np.interp to interpolate
                        Pos_x = np.interp(time, t_values, x_values)
                        Pos_y = np.interp(time, t_values, y_values)
                        Pos_z = np.interp(time, t_values, z_values)

                        Vel_x = np.interp(time, t_values, vx_values)
                        Vel_y = np.interp(time, t_values, vy_values)
                        Vel_z = np.interp(time, t_values, vz_values)







                    Pos_Radar=ECEF2Local_NWU([Pos_x,Pos_y,Pos_z], Sensor_Lat, Sensor_Long, Sensor_Height, a, e2, 0)
                    Vel_Radar =ECEF2Local_NWU([Vel_x, Vel_y, Vel_z], Sensor_Lat, Sensor_Long, Sensor_Height, a, e2, 1)
                    # convert to NWU RADAR Coorditanes system

                    enu_pos=nwu2enu(np.array(Pos_Radar))
                    enu_vel = nwu2enu(np.array(Vel_Radar))

                    [White_Range,White_Az,White_El,White_Dop]=enu2spherical_rdot (enu_pos,enu_vel)  # radians

                    az_diff_deg=math.degrees(Calc_dAlpha(White_Az, math.radians(Center_Az)))

                    az_cond=abs(az_diff_deg) <=  AzWidth
                    el_cond1=math.degrees(White_El) >= MinElevation  # degrees
                    el_cond2=math.degrees(White_El) <= MaxElevation
                    range_cond1=White_Range >= MinRange
                    range_cond2 = White_Range <= MaxRange

                    fov_cond =(az_cond and
                   el_cond1 and el_cond2 and
                   range_cond1 and range_cond2)
                    if fov_cond:
                        loss_mat=0
                        tgt_snr_db = snr_estimation_DAIR(White_Range, loss_mat, Params)
                        Pd = 1
                        if tgt_snr_db < 0:  # Target is too far away
                            Pd = 0
                        else:
                            if tgt_snr_db < getParam(Params, 'AT_Sys_Params.max_snr_pd_est'): # High SNR will have a Pd of 1
                                if getParam(Params, 'AT_Sys_Params.swerling_type') == 1:

                                    Pd = Pd_case1(tgt_snr_db, getParam(Params, 'AT_Sys_Params.pfa'))

                                else:

                                    Pd = Pd_case3(tgt_snr_db, getParam(Params, 'AT_Sys_Params.pfa'))


                            Rand_Num = np.random.rand()  # Generates a random float in [0,1]

                            prob_cond = Rand_Num <= Pd  # Pd should be a boolean or array of booleans

                            # Use element-wise logical AND if Pd is an array

                            if np.any(fov_cond & prob_cond):  # Use element-wise AND for arrays
                                    # Your code here

                                    # Initialize This_Plot as a dictionary



                                    # Create an empty DataFrame

                                    This_Plot = pd.DataFrame()

                                    # Assign values
                                    This_Plot.loc[0, "Time"] = time
                                    This_Plot.loc[0, "SimId"] =sgid
                                    This_Plot['SimId']=This_Plot["SimId"].astype(int)


                                    This_Plot.loc[0, "SensorId"] = This_Sensor.SensorId
                                    This_Plot.loc[0, "SensorType"] = This_Sensor.SensorType
                                    This_Plot["SensorId"]=This_Plot["SensorId"].astype(int)
                                    ErrorFactor = getParam(Params,
                                                           'AT_Sys_Params.Error_Factor')  # Convert string to float

                                    This_Plot.loc[0, "AzError"] = np.radians(
                                        This_Sensor.AzError)  # Convert degrees to radians

                                    This_Plot.loc[0, "Az"] = White_Az + ErrorFactor * This_Plot.loc[
                                        0, "AzError"] * np.random.randn() + This_Sensor.AzBias
                                    if Last_Plot.empty:
                                        This_Plot['Az_Rate']=0
                                        This_Plot['Az_Rate_Error'] = BearingRateError
                                    else:
                                        This_Plot['Az_Rate']=(This_Plot['Az']-Last_Plot['Az'])/(This_Plot['Time']-Last_Plot['Time'])
                                        This_Plot['Az_Rate_Error']=BearingRateError



                                    if int(This_Sensor['SensorType'])==1:


                                        This_Plot.loc[0, "RangeError"] = This_Sensor.RangeError

                                        This_Plot.loc[0, "ElError"] = np.radians(This_Sensor.ElError)
                                        This_Plot.loc[0, "DopplerError"] = This_Sensor.DopplerError

                                        # Error factor from UI (assuming it's a string that needs conversion)


                                        # Compute values with error terms
                                        This_Plot.loc[0, "Range"] = White_Range + ErrorFactor * This_Plot.loc[
                                            0, "RangeError"] * np.random.randn() + This_Sensor.RangeBias

                                        This_Plot.loc[0, "El"] = White_El + ErrorFactor * This_Plot.loc[
                                            0, "ElError"] * np.random.randn() + This_Sensor.ElBias
                                        This_Plot.loc[0, "Doppler"] = White_Dop + ErrorFactor * This_Plot.loc[
                                            0, "DopplerError"] * np.random.randn() + This_Sensor.DopplerBias

                                        This_Plot, H = Shere2Cart(This_Plot)

                                        # Convert Local NWU to ECEF

                                        ecef_pos = Local_NWU2ECEF(
                                            np.array(
                                                [This_Plot.loc[0, "x"], This_Plot.loc[0, "y"], This_Plot.loc[0, "z"]]),
                                            This_Sensor.Lat, This_Sensor.Long, This_Sensor.Height,
                                            a, e2, 0
                                        )
                                        Lat, Long, Alt = pm.ecef2geodetic(ecef_pos[0, 0], ecef_pos[0, 1],
                                                                          ecef_pos[0, 2])

                                        # Store final location values
                                        This_Plot.loc[0, "Lat"] = Lat  # Convert radians to degrees
                                        This_Plot.loc[0, "Long"] = Long
                                        This_Plot.loc[0, "Alt"] = Alt

                                        This_Plot.loc[0, "ECEF_X"] = ecef_pos[0, 0]
                                        This_Plot.loc[0, "ECEF_Y"] = ecef_pos[0, 1]
                                        This_Plot.loc[0, "ECEF_Z"] = ecef_pos[0, 2]



                                    else:
                                        This_Plot.loc[0, "RangeError"] = np.nan

                                        This_Plot.loc[0, "ElError"] = np.nan
                                        This_Plot.loc[0, "DopplerError"] = np.nan

                                        # Error factor from UI (assuming it's a string that needs conversion)


                                        # Compute values with error terms
                                        This_Plot.loc[0, "Range"] = np.nan

                                        This_Plot.loc[0, "El"] = np.nan
                                        This_Plot.loc[0, "Doppler"] = np.nan


                                        This_Plot.loc[0, "Lat"] = np.nan  # Convert radians to degrees
                                        This_Plot.loc[0, "Long"] = np.nan
                                        This_Plot.loc[0, "Alt"] = np.nan

                                        This_Plot.loc[0, "ECEF_X"] = np.nan
                                        This_Plot.loc[0, "ECEF_Y"] =np.nan
                                        This_Plot.loc[0, "ECEF_Z"] = np.nan

                                        This_Plot['x'] = np.nan
                                        This_Plot['y'] = np.nan
                                        This_Plot['z'] = np.nan

                                        This_Plot["CM_Cart"] = None
                                        This_Plot = This_Plot.astype({"CM_Cart": "object"})
                                        This_Plot.at[0, "CM_Cart"] = [np.nan] * 9


                                    This_Plot["CM_Polar"]=None
                                    This_Plot = This_Plot.astype({"CM_Polar": "object"})
                                    This_Plot.at[0,"CM_Polar"] = np.diag(np.array([This_Plot['RangeError'].iloc[0]**2,This_Plot['AzError'].iloc[0]**2,This_Plot['ElError'].iloc[0]**2,
                                                                                This_Plot["DopplerError"].iloc[0]**2,BearingRateError**2,0])).flatten().tolist()
                                    # df_combined = pd.concat([df1, df2], axis=1)
                                    This_Plot=pd.concat ([This_Plot.reset_index(drop=True),This_EPL.reset_index(drop=True)],axis=1)



                                    # Now concatenate horizontally

                                    Last_Plot = This_Plot
                                    Drone_Plots = pd.concat([Drone_Plots, This_Plot])







            Drone_Plots = Drone_Plots.sort_values(by=["Time", "SensorId","SimId"], ascending=[True, True,True])
        print('Completed Building Plots ')
        return Drone_Plots
    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        return None


def Drones_Tracker (ScenarioFolder,Params,Sensors,Drone_Plots):
    import numpy as np
    import pandas as pd
    from my_subscript import Local_NWU2ECEF,getParam,Local_NWU2ECEF_CovMat,ECEF2Local_NWU,ECEF2Local_NWU_CovMat,Predict_Track,ConvertCartesian2Sphere,Calc_dAlpha
    import traceback

    import warnings
    warnings.filterwarnings("ignore")

    try:
        a = getParam(Params, 'AT_Sys_Params.a')
        e2 = getParam(Params, 'AT_Sys_Params.e2')
        NoiseModel = getParam(Params, 'AT_Sys_Params.NoiseModel')

        Ref_Point_Lat = getParam(Params, 'AT_Sys_Params.Ref_Point_Lat')
        Ref_Point_Long = getParam(Params, 'AT_Sys_Params.Ref_Point_Long')
        Ref_Point_Alt = getParam(Params, 'AT_Sys_Params.Ref_Point_Alt')

        alpha= getParam(Params, 'AT_Sys_Params.alpha')
        SigmaM = getParam(Params, 'AT_Sys_Params.SigmaM')
        MaxChi2ToCorr=getParam(Params, 'AT_Sys_Params.MaxChi2ToCorr')
        Track_Kill_Time=getParam(Params, 'AT_Sys_Params.Track_Kill_Time')
        Track_Extrapolate_Time=getParam(Params, 'AT_Sys_Params.Track_Extrapolate_Time')


        Drone_Tracks = pd.DataFrame()
            # ({
            # 'Time': [],
            # 'ID': [],
            # 'SimId': [],
            # 'SensorID': [],
            # 'Pos': [],
            # 'Vel': [],
            # 'Acc'"[]"
            # 'CM': [],
            # 'Lat': [],
            # 'Long': [],
            # 'Alt': [],
            # 'SV_Ref': [],
            # 'CM_Ref': []}))



        for index,row in Drone_Plots.iterrows():
            This_Plot=row
           #  print(This_Plot['Time'], This_Plot['SensorId'], This_Plot['SimId'])
            This_Radar = Sensors[Sensors['SensorId'] == This_Plot.SensorId]

            if Drone_Tracks.empty:
                This_Radar_Tracks = pd.DataFrame()
            else:
                This_Radar_Tracks = Drone_Tracks[Drone_Tracks['SensorId'] == This_Plot['SensorId']]

            if len(This_Radar_Tracks)==0:
                New_Track = Create_New_Drones_Track(This_Plot, This_Radar, Params)

                # Update the ID and SimID
                New_Track['ID'] = 1 + This_Plot.SensorId * 1000
                # New_Track.to_excel(ScenarioFolder + "temp1.xlsx", index=False)





                # Append to  DataFrame
                Drone_Tracks = pd.concat([Drone_Tracks, New_Track], ignore_index=True)
                Drone_Tracks.to_excel(ScenarioFolder + "Drone_Tracks.xlsx", index=False)

            else:
                IDs = This_Radar_Tracks["ID"].unique()
                Cand_Pred_SV = []
                Cand_Pred_CM = []
                Cand_Innnov = []
                Cand_Innnov_Mat = []
                Cand_Chi2 = []
                Cand_Gains = []
                Cand_H = []
                Cand_IDs=[]

                for k in range(len(IDs)):
                    filtered = This_Radar_Tracks[
                        (This_Radar_Tracks["ID"] == IDs[k]) &
                        (This_Radar_Tracks['Time'] < This_Plot['Time'])
                        ]

                    if not filtered.empty:
                        Corr_Candidate = filtered.iloc[-1]
                    else:

                        continue    # or handle it another way

                    # MATLAB max(find(...)) equivalent

                    if Corr_Candidate["Alive"]:
                        dT = This_Plot.Time - Corr_Candidate.Time

                        if dT == 0:
                            Cand_Chi2.append(float("inf"))
                            continue

                        if NoiseModel == "'Linear'":
                            SV_Prev = np.hstack((Corr_Candidate.Pos, Corr_Candidate.Vel))
                            CM_Prev = np.reshape(Corr_Candidate.CM, (6, 6))

                            F = np.array([
                                [1, 0, 0, dT, 0, 0],
                                [0, 1, 0, 0, dT, 0],
                                [0, 0, 1, 0, 0, dT],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]
                            ])

                            SV_Pred = F @ SV_Prev
                            Qv = 10
                            Qp = Qv * dT
                            Q = np.diag([Qp ** 2, Qp ** 2, Qp ** 2, Qv ** 2, Qv ** 2, Qv ** 2])
                            CM_Pred = F @ CM_Prev @ F.T + Q

                        elif NoiseModel == "'Singer'":
                            SV_Prev = np.hstack((Corr_Candidate.Pos, Corr_Candidate.Vel, Corr_Candidate.Acc))
                            CM_Prev = np.reshape(Corr_Candidate.CM, (9, 9))

                            SV_Pred, CM_Pred = Predict_Track(SV_Prev, CM_Prev, dT, alpha,
                                                             SigmaM)

                        if not np.all(np.isreal(np.linalg.eigvals(CM_Prev))):
                            continue

                        if not np.all(np.isreal(np.linalg.eigvals(CM_Pred))):
                            continue

                        SV_Pred = SV_Pred.reshape(-1, 1)   #9x1



                        SV_Pred = np.transpose(SV_Pred)
                        # CM_Pred=CM_Pred[0:6,0:6]
                        SV_Sphere, CM_Sphere, H = ConvertCartesian2Sphere(SV_Pred[0][0:6], CM_Pred[0:6,0:6])  # colums



                        innovation_vector = np.array([
                            This_Plot.Range - SV_Sphere[0,0],
                            Calc_dAlpha(This_Plot.Az, SV_Sphere[0,1]),
                            Calc_dAlpha(This_Plot.El, SV_Sphere[0,2]),
                            This_Plot.Doppler - SV_Sphere[0,3]
                        ])

                        CM_Sphere_Plot = np.diag([
                            This_Plot.RangeError ** 2,
                            This_Plot.AzError ** 2,
                            This_Plot.ElError ** 2,
                            This_Plot.DopplerError ** 2
                        ])

                        innovation_matrix = H @ CM_Pred[:6, :6] @ H.T + CM_Sphere_Plot
                        Filter_Gain = CM_Pred[:6, :6] @ H.T @ np.linalg.inv(innovation_matrix)

                        Cand_Pred_SV.append(SV_Pred.tolist())

                        if NoiseModel == "'Linear'":
                            Cand_Pred_CM.append(CM_Pred.flatten().tolist())
                        elif NoiseModel == "'Singer'":
                            Cand_Pred_CM.append(CM_Pred.flatten().tolist())

                        Cand_Innnov.append(innovation_vector.tolist())
                        Cand_Innnov_Mat.append(innovation_matrix.flatten().tolist())

                        Cand_Chi2.append(float(innovation_vector.T @ np.linalg.inv(innovation_matrix) @ innovation_vector))
                        Cand_Chi2_NoDop = float(
                            innovation_vector[:3].T @ np.linalg.inv(innovation_matrix[:3, :3]) @ innovation_vector[:3])

                        Cand_Gains.append(Filter_Gain.flatten().tolist())
                        Cand_H.append(H.flatten().tolist())
                        Cand_IDs.append(IDs[k])

                    else:
                        pass
                        # Cand_Chi2.append(float("inf"))  # Dead track will never win
                        #   TBD this code was not examined
                if len(Cand_Chi2) == 0:
                    New_Track = Create_New_Drones_Track(This_Plot, This_Radar, Params)
                    New_Track.ID = max(This_Radar_Tracks['ID']) + 1


                    try:

                        Drone_Tracks = pd.concat([Drone_Tracks, New_Track], ignore_index=True)

                    except Exception as err:
                        print(f"Error: {err}")
                        traceback.print_exc()


                        Drone_Tracks.to_excel(ScenarioFolder + "Old.xlsx", index=False)
                        New_Track.to_excel(ScenarioFolder + "New.xlsx", index=False)


                        a=1

                else:

                    Min_Chi2 = np.min(Cand_Chi2)
                    ind = np.argmin(Cand_Chi2)

                    if Min_Chi2 <= MaxChi2ToCorr:
                        Best_Gain = np.reshape(Cand_Gains[ind], (6, 4))
                        Best_Innovation_SV = Cand_Innnov[ind]
                        Best_Innovation_CM = np.reshape(Cand_Innnov_Mat[ind], (4, 4))
                        Best_Pred_SV = Cand_Pred_SV[ind]

                        if NoiseModel == "'Linear'":
                            Best_Pred_CM = np.reshape(Cand_Pred_CM[ind], (6, 6))
                        elif NoiseModel == "'Singer'":
                            Best_Pred_CM = np.reshape(Cand_Pred_CM[ind], (9, 9))

                        Best_H = np.reshape(Cand_H[ind], (4, 6))
                        Delta_SV = np.dot(Best_Gain, np.array(Best_Innovation_SV).T)

                        if NoiseModel == "'Singer'":
                            Delta_SV=np.vstack((Delta_SV.reshape(-1, 1), np.array([[0],[0],[0]])))


                        StateVecUpdate = np.array(Best_Pred_SV).reshape(-1,1) + Delta_SV
                        CovMatUpdate1 = (np.eye(6) - np.dot(Best_Gain, Best_H)) @ Best_Pred_CM[:6, :6]

                        if NoiseModel == "'Singer'":
                            CovMatUpdate = np.zeros((9, 9))

                            # Place matrix1 in the top-left corner
                            CovMatUpdate[:6, :6] = CovMatUpdate1
                            CovMatUpdate[6:9, 6:9] = Best_Pred_CM[6:9,6:9]
                        else:
                            CovMatUpdate=CovMatUpdate1





                        if not np.all(np.isreal(np.linalg.eigvals(CovMatUpdate))):
                            New_Track = Create_New_Drones_Track(This_Plot, This_Radar, Params)
                            New_Track.ID =max(This_Radar_Tracks['ID'])+1


                            Drone_Tracks = pd.concat([Drone_Tracks, pd.DataFrame([New_Track])], ignore_index=True)

                        else:
                            Updated_Track = pd.DataFrame(
                                columns=['Alive', 'SensorId', 'SimId', 'Time', 'Pos', 'Vel', 'Acc', 'Lambda', 'Speed',
                                         'Heading', 'CM', 'Lat', 'Long', 'Alt', 'Pos_Ref', 'Vel_Ref','CM_Ref'])

                            Updated_Track.at[0,'ID'] = int(Cand_IDs[ind])
                            Updated_Track.at[0,'SimId'] =int(This_Plot.SimId)
                            Updated_Track.at[0,'Time'] = float(This_Plot.Time)
                            Updated_Track.at[0,'Alive'] = int(1)
                            Updated_Track.at[0,'SensorId'] = int(This_Plot.SensorId)
                            Updated_Track.at[0,'Pos'] = (np.array(StateVecUpdate[0:3]).reshape(1,3)).tolist()[0]
                            Updated_Track.at[0, 'Vel'] = (np.array(StateVecUpdate[3:6]).reshape(1, 3)).tolist()[0]
                            Updated_Track.at[0, 'Acc'] = (np.array(StateVecUpdate[6:9]).reshape(1, 3)).tolist()[0]
                            Heading=[np.arctan2(-Updated_Track.at[0, 'Vel'][1], Updated_Track.at[0, 'Vel'][0]) * 180 / np.pi][0]
                            if Heading<0:
                                Heading=Heading+360
                            Updated_Track.at[0,'Heading'] = Heading
                            Updated_Track.at[0,'Speed'] = float(np.linalg.norm(Updated_Track.at[0,'Vel']))


                            # SV_Cart = np.hstack((Updated_Track.at[0,'Pos'][0], Updated_Track.at[0,'Vel'][0]))

                            if NoiseModel == "'Linear'":
                                Updated_Track.at[0,'CM'] = [CovMatUpdate.flatten()]
                            elif NoiseModel == "'Singer'":


                                # Updated_Track.at[0,'CM'] = (np.array(CovMatUpdate[6:9]).reshape(1, 3)).tolist()
                                Updated_Track.at[0,'CM']=CovMatUpdate.tolist()


                                ECEF_Pos = Local_NWU2ECEF(Updated_Track["Pos"][0], This_Radar.Lat, This_Radar.Long,This_Radar.Height, a, e2,0)
                                ECEF_Vel = Local_NWU2ECEF(Updated_Track["Vel"][0],This_Radar.Lat, This_Radar.Long,This_Radar.Height, a, e2,1)

                                lat, long, alt = pm.ecef2geodetic(ECEF_Pos[0, 0], ECEF_Pos[0, 1], ECEF_Pos[0, 2])

                                Updated_Track.at[0,'Lat'] = float(lat)
                                Updated_Track.at[0,'Long'] = float(long)
                                Updated_Track.at[0,'Alt'] = float(alt)

                                # Convert position and velocity to Local NWU coordinates
                                Local_Pos = ECEF2Local_NWU(ECEF_Pos, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2,
                                                           0)
                                Local_Vel = ECEF2Local_NWU(ECEF_Vel, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2,
                                                           1)

                                # Update SV_Ref and CM_Ref in Updated_Track
                                Updated_Track.at[0,'Pos_Ref']= np.array(Local_Pos).reshape(1,3).tolist()

                                Updated_Track.at[0, 'Vel_Ref'] = np.array(Local_Vel).reshape(1, 3).tolist()





                                # Noise Model Handling
                                if NoiseModel == 'Linear':
                                    Local_CM_Radar = np.reshape(Updated_Track['CM'], (6, 6))
                                elif NoiseModel == "'Singer'":
                                    Local_CM_Radar = np.array(Updated_Track['CM'].iloc[0]).reshape(9, 9)

                                Cart_SV = np.array(Updated_Track.at[0, 'Pos'] + Updated_Track.at[0, 'Vel'])
                                Cart_CM=Local_CM_Radar[0:6,0:6]
                                SV_Sphere, CM_Sphere, H = ConvertCartesian2Sphere(Cart_SV,Cart_CM)
                                Updated_Track.at[0, 'RangeErr']=np.sqrt(CM_Sphere[0,0])
                                Updated_Track.at[0, 'AzErr'] = np.sqrt(CM_Sphere[1, 1])*180 / np.pi
                                Updated_Track.at[0, 'ElErr'] = np.sqrt(CM_Sphere[2, 2])*180 / np.pi


                                # Convert covariance matrix from NWU to ECEF
                                ECEF_CM = Local_NWU2ECEF_CovMat(Local_CM_Radar[:6, :6], This_Radar.Lat,
                                                                This_Radar.Long)

                                Updated_Track['ECEF_Pos'] = None
                                Updated_Track.at[0, 'ECEF_Pos'] = ECEF_Pos.flatten().tolist()

                                Updated_Track['ECEF_Vel'] = None
                                Updated_Track.at[0, 'ECEF_Vel'] = ECEF_Vel.flatten().tolist()



                                Updated_Track['ECEF_CM'] = None

                                Updated_Track.at[0, 'ECEF_CM'] = ECEF_CM.flatten().tolist()








                                # Convert covariance matrix to Local NWU
                                Local_CM = ECEF2Local_NWU_CovMat(ECEF_CM, Ref_Point_Lat, Ref_Point_Long)

                                Updated_Track.at[0, 'CM_Ref'] = [Local_CM.tolist()]
                                ###############################################

                                B = np.linalg.det(2 * np.pi * Best_Innovation_CM)
                                C = -0.5 *np.array(Best_Innovation_SV) @ np.linalg.inv(Best_Innovation_CM) @ np.array(Best_Innovation_SV).T
                                Updated_Track.at[0,'Lambda'] = (1 / np.sqrt(B)) * np.exp(C)
                                Updated_Track.at[0, 'Extrapolated'] = 0



                                Drone_Tracks = pd.concat([Drone_Tracks, Updated_Track], ignore_index=True)
                    else:
                        New_Track = Create_New_Drones_Track(This_Plot, This_Radar, Params)
                        New_Track.ID = max(This_Radar_Tracks['ID']) + 1
                        Drone_Tracks = pd.concat([Drone_Tracks, New_Track], ignore_index=True)
            # Kill track that are still alive but not updated for long time
            for unique_id in Drone_Tracks["ID"].unique():
                last_index = Drone_Tracks[
                    (Drone_Tracks["ID"] == unique_id) & (Drone_Tracks["Extrapolated"] ==0 )
                    ].index[-1]
                Last_Update = pd.DataFrame([Drone_Tracks.loc[last_index]])

                last_index1 = Drone_Tracks[
                    (Drone_Tracks["ID"] == unique_id)
                    ].index[-1]
                Last_Record = pd.DataFrame([Drone_Tracks.loc[last_index1]])
                if Last_Record['Alive'].iloc[0]:
                    dT=This_Plot['Time']-Last_Update['Time'].iloc[0]
                    if dT<=Track_Extrapolate_Time:
                        pass
                    else:
                        # Last_Update['Pos'].iloc[0] = (np.array(Last_Update['Pos'].iloc[0]) + np.array(
                        #     Last_Update[
                        # 'Vel'].iloc[0]) * dT + np.array(
                        #    Last_Update['Acc'].iloc[0]) * dT ** 2 / 2).tolist()



                        #####################
                        vel = np.array(Last_Update.iloc[0]['Vel'])
                        acc = np.array(Last_Update.iloc[0]['Acc'])
                        pos = np.array(Last_Update.iloc[0]['Pos'])

                        # Update velocity and position
                        vel += acc * dT
                        pos += vel * dT + acc * dT ** 2 / 2

                        idx = Last_Update.index[0]  # Get the actual index (e.g., 96)

                        Last_Update.at[idx, 'Vel'] = vel.tolist()
                        Last_Update.at[idx, 'Pos'] = pos.tolist()


                        ##################################################



                        Last_Update['Extrapolated'] = 1

                        Last_Update['Time'] = This_Plot['Time']
                        This_Radar = Sensors[Sensors['SensorId'] == Last_Update['SensorId'].iloc[0]]

                        ECEF_Pos = Local_NWU2ECEF(Last_Update["Pos"].iloc[0], This_Radar.Lat, This_Radar.Long,
                                                  This_Radar.Height, a, e2, 0)
                        ECEF_Vel = Local_NWU2ECEF(Last_Update["Vel"].iloc[0], This_Radar.Lat, This_Radar.Long,
                                                  This_Radar.Height, a, e2, 1)



                        lat, long, alt = pm.ecef2geodetic(ECEF_Pos[0, 0], ECEF_Pos[0, 1], ECEF_Pos[0, 2])

                        Last_Update['Lat'] = float(lat)
                        Last_Update['Long'] = float(long)
                        Last_Update['Alt'] = float(alt)

                        # Convert position and velocity to Local NWU coordinates
                        Local_Pos = ECEF2Local_NWU(ECEF_Pos, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2,
                                                   0)
                        Local_Vel = ECEF2Local_NWU(ECEF_Vel, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2,
                                                   1)

                        # Update SV_Ref and CM_Ref in Updated_Track
                        Last_Update['Pos_Ref'] = np.array(Local_Pos).reshape(1, 3).tolist()

                        Last_Update['Vel_Ref'] = np.array(Local_Vel).reshape(1, 3).tolist()

                        if dT>=Track_Kill_Time:
                            Last_Update['Alive'] = 0

                        Drone_Tracks = pd.concat([Drone_Tracks, Last_Update], ignore_index=True)
                        pass



        return Drone_Tracks










    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        print (This_Plot['Time'],This_Plot['SensorId'],This_Plot['SimId'])
        return None




def Create_New_Drones_Track (This_Plot, This_Radar, Params):

    from my_subscript import Local_NWU2ECEF, getParam, Local_NWU2ECEF_CovMat, ECEF2Local_NWU, ECEF2Local_NWU_CovMat,ConvertSphere2Cartesian,ConvertCartesian2Sphere
    import numpy as np
    try:
        # Read parameters
        from my_subscript import getParam
        import traceback
        NoiseModel = getParam(Params, 'AT_Sys_Params.NoiseModel')
        Init_Pos_Err=getParam(Params, 'AT_Sys_Params.Init_Pos_Err')
        Init_Vel_Err = getParam(Params, 'AT_Sys_Params.Init_Vel_Err')
        Init_Acc_Err = getParam(Params, 'AT_Sys_Params.Init_Acc_Err')
        a = getParam(Params, 'AT_Sys_Params.a')
        e2 = getParam(Params, 'AT_Sys_Params.e2')
        Ref_Point_Lat = getParam(Params, 'AT_Sys_Params.Ref_Point_Lat')
        Ref_Point_Long = getParam(Params, 'AT_Sys_Params.Ref_Point_Long')
        Ref_Point_Alt = getParam(Params, 'AT_Sys_Params.Ref_Point_Alt')


        # Initialize state vector and covariance matrix
        New_Track = pd.DataFrame(columns=['Alive','SensorId','SimId','Time', 'Pos','Vel','Acc','Lambda','Speed','Heading','CM','Lat','Long','Alt','Pos_Ref','Vel_Ref','CM_Ref','ID'])
        New_Track.at[0, 'Alive'] = 1
        New_Track.at[0, 'Time'] = This_Plot['Time']
        New_Track.at[0, 'SensorId'] = int(This_Plot['SensorId'])
        New_Track.at[0,'SimId'] = int(This_Plot.SimId)
        SV_Sphere = np.array([This_Plot['Range'], This_Plot['Az'], This_Plot['El'], This_Plot['Doppler'], 0, 0])
        CM_Sphere = np.diag([This_Plot['RangeError'] ** 2, This_Plot['AzError'] ** 2, This_Plot['ElError'] ** 2,
                             This_Plot['DopplerError'] ** 2, 1, 1])

        # Convert to Cartesian coordinates
        SV_Cart, CM_Cart = ConvertSphere2Cartesian(SV_Sphere, CM_Sphere)

        # Initialize new track


        x=SV_Cart[0,0]
        y=SV_Cart[0,1]
        z=SV_Cart[0,2]
        New_Track.at[0, 'Pos'] =  [x, y, z]  # Store as a list


        # New_Track['Pos'] = SV_Cart[:3]
        New_Track.at[0,'Vel'] = [0, 0, 0] # Initial velocity set to zero
        Speed = New_Track['Vel'].apply(lambda vel: np.linalg.norm(np.array(vel)))
        New_Track.at[0, 'Speed']=float(Speed)


        New_Track.at[0,'Acc'] = [0, 0, 0]  # Acceleration set to zero

        # Update state vector

        # Lambda, speed, and heading
        New_Track.at[0,'Lambda'] = 0.5





        # Vel=np.array(New_Track['Vel'])

        Heading = np.array(New_Track['Vel'].apply(lambda vel: np.arctan2(-New_Track.at[0,'Vel'][1], New_Track.at[0,'Vel'][0]) * 180 / np.pi))
        if Heading<0:
            Heading=Heading+360
        New_Track.at[0,'Heading'] = float(Heading)

        # Handle different models (Singer or Linear)
        if  NoiseModel == "'Singer'":
            CM_Cart = np.diag([Init_Pos_Err ** 2,Init_Pos_Err ** 2,Init_Pos_Err ** 2,
                               Init_Vel_Err ** 2,Init_Vel_Err ** 2,Init_Vel_Err ** 2,
                               Init_Acc_Err ** 2,Init_Acc_Err ** 2,Init_Acc_Err ** 2])

            # New_Track.at[0,'CM'] = tuple(CM_Cart.reshape(1, 81))
            New_Track.at[0, 'CM'] = CM_Cart.tolist()

            # Convert from Local NWU to ECEF for position and velocity
            # Pos=New_Track.loc[0, "Pos"]

            ECEF_Pos = Local_NWU2ECEF(New_Track.loc[0, "Pos"], This_Radar.Lat, This_Radar.Long, This_Radar.Height, a, e2,
                                      0)

            ECEF_Vel = Local_NWU2ECEF(New_Track.loc[0, "Vel"], This_Radar.Lat,
                                      This_Radar.Long, This_Radar.Height, a, e2, 1)

            # Noise Model Handling
            if NoiseModel == 'Linear':
                Local_CM_Radar = np.reshape(New_Track['CM'], (6, 6))
            elif NoiseModel == "'Singer'":
                Local_CM_Radar = np.array(New_Track['CM'].iloc[0]).reshape(9, 9)

            # Convert covariance matrix from NWU to ECEF
            ECEF_CM = Local_NWU2ECEF_CovMat(Local_CM_Radar[:6, :6], This_Radar.Lat,
                                            This_Radar.Long)

            New_Track['ECEF_Pos'] = None
            New_Track.at[0, 'ECEF_Pos'] = ECEF_Pos.flatten().tolist()

            New_Track['ECEF_Vel'] = None
            New_Track.at[0, 'ECEF_Vel'] = ECEF_Vel.flatten().tolist()


            New_Track['ECEF_CM'] = None


            New_Track.at[0, 'ECEF_CM'] = ECEF_CM.flatten().tolist()



            # Convert position to Lat, Longg, Alt
            lat, long, alt = pm.ecef2geodetic(ECEF_Pos[0, 0], ECEF_Pos[0, 1], ECEF_Pos[0, 2])

            New_Track['Lat'] = lat
            New_Track['Long'] = long
            New_Track['Alt'] = alt

            # Convert position and velocity to Local NWU coordinates
            Local_Pos = ECEF2Local_NWU(ECEF_Pos, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2, 0)
            Local_Vel = ECEF2Local_NWU(ECEF_Vel, Ref_Point_Lat, Ref_Point_Long, Ref_Point_Alt, a, e2, 1)

            New_Track.at[0, 'Pos_Ref'] = np.array(Local_Pos).reshape(1, 3).tolist()

            New_Track.at[0, 'Vel_Ref'] = np.array(Local_Vel).reshape(1, 3).tolist()

            # Update SV_Ref and CM_Ref in New_Track




            # Convert covariance matrix to Local NWU
            Local_CM = ECEF2Local_NWU_CovMat(ECEF_CM, Ref_Point_Lat, Ref_Point_Long)

            Cart_SV = np.array(New_Track.at[0, 'Pos'] + New_Track.at[0, 'Vel'])
            Cart_CM = CM_Cart[0:6, 0:6]
            SV_Sphere, CM_Sphere, H = ConvertCartesian2Sphere(Cart_SV, Cart_CM)
            New_Track.at[0, 'RangeErr'] = np.sqrt(CM_Sphere[0, 0])
            New_Track.at[0, 'AzErr'] = np.sqrt(CM_Sphere[1, 1])*180 / np.pi
            New_Track.at[0, 'ElErr'] = np.sqrt(CM_Sphere[2, 2])*180 / np.pi


            # New_Track.at[0,'CM_Ref'] = tuple([Local_CM.flatten()])
            New_Track.at[0, 'CM_Ref'] = [Local_CM.tolist()]
            New_Track.at[0, 'Extrapolated'] = 0


            # New_Track['Acc'] = np.random.randn(3)  # Uncomment if you want to randomize Acc

        elif NoiseModel == 'Linear':
            CM_Cart =  np.diag([Init_Pos_Err ** 2,Init_Pos_Err ** 2,Init_Pos_Err ** 2,
                               Init_Vel_Err ** 2,Init_Vel_Err ** 2,Init_Vel_Err ** 2])

            New_Track.at[0,'CM'] = CM_Cart.reshape(1, 36)

        # Convert back to spherical coordinates and get error estimates
        # SV_Sphere, CM_Sphere, H = ConvertCartesian2Sphere(SV_Cart[:6], CM_Cart[:6, :6])

        # New_Track['Range'] = SV_Sphere[0]
        # New_Track['Error_Azimuth'] = math.sqrt(CM_Sphere[1, 1]) * 180 / np.pi
        # New_Track['Error_Elevation'] = math.sqrt(CM_Sphere[2, 2]) * 180 / np.pi

        # Radar info
        # New_Track['Radar_Lat'] = This_Radar['Lat']
        # New_Track['Radar_Long'] = This_Radar['Long']
        # New_Track['Radar_Height'] = This_Radar['Height']

    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        return None

    return New_Track

def Export_Tracks_To_TCU(Drone_Tracks, Sensors):
    import traceback
    import pandas as pd
    try:
        TCU_Tracks_Table=pd.DataFrame()
        import datetime

        # Get today's date
        today = datetime.date.today()

        # Combine today's date with midnight time (00:00:00)
        midnight = datetime.datetime.combine(today, datetime.time())

        # Calculate the epoch time (seconds since January 1, 1970)
        epoch_time = int((midnight - datetime.datetime(1970, 1, 1)).total_seconds())
        T0_Microsec =epoch_time*1e6

        Drone_Tracks=Drone_Tracks[Drone_Tracks['Alive']==1]
        TCU_Tracks_Table['Object_Id']=Drone_Tracks['ID']
        TCU_Tracks_Table['Sensor_Id'] = Drone_Tracks['SensorId']
        TCU_Tracks_Table['Measurement_Time'] = Drone_Tracks['Time'] * 1e6 + T0_Microsec

        TCU_Tracks_Table['ECEF_Pos'] = Drone_Tracks['ECEF_Pos']
        TCU_Tracks_Table['ECEF_Vel'] = Drone_Tracks['ECEF_Vel']
        TCU_Tracks_Table['ECEF_CM'] = Drone_Tracks['ECEF_CM']

        Radar_Ids = Drone_Tracks['SensorId'].unique()

        # Iterate over each Radar_Id
        for radar_id in Radar_Ids:
            # Get the matching radar data from Sensors DataFrame
            this_radar = Sensors[Sensors['SensorId'] == radar_id]

            # Find indices in Drones_Tracks where SensorId matches
            ind = Drone_Tracks[Drone_Tracks['SensorId'] == radar_id].index

            # Assign corresponding values to TCU_Tracks_Table
            TCU_Tracks_Table.loc[ind, 'Sensor_Lat'] = this_radar['Lat'].values[0]
            TCU_Tracks_Table.loc[ind, 'Sensor_Long'] = this_radar['Long'].values[0]
            TCU_Tracks_Table.loc[ind, 'Sensor_Height'] = this_radar['Height'].values[0]
            return TCU_Tracks_Table

    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        return None





    a=1

def Export_Sigint_To_TCU(Sigint_Plots, Sigint_Sensors,EPL,Params):
    import traceback
    import pandas as pd
    from my_subscript import getParam
    pd.set_option('display.float_format', '{:.10f}'.format)
    try:

        BearingRateError= getParam(Params, 'AT_Sys_Params.BearingRateError')
        TCU_Sigint_Table = pd.DataFrame()
        import datetime

        # Get today's date
        today = datetime.date.today()

        # Combine today's date with midnight time (00:00:00)
        midnight = datetime.datetime.combine(today, datetime.time())

        # Calculate the epoch time (seconds since January 1, 1970)
        epoch_time = int((midnight - datetime.datetime(1970, 1, 1)).total_seconds())
        T0_Microsec = epoch_time * 1e6



        TCU_Sigint_Table['Measurment_Time'] = Sigint_Plots['Time'] * 1e6 + T0_Microsec
        TCU_Sigint_Table['Sensor_Id'] = Sigint_Plots['SensorId']
        TCU_Sigint_Table['Bearing'] = Sigint_Plots['Az']

        TCU_Sigint_Table['Bearing_Rate'] = Sigint_Plots['Az_Rate']


        TCU_Sigint_Table[['Bearing_Var', 'Bearing_Rate_Var']] = Sigint_Plots['CM_Polar'].apply(lambda x: pd.Series([x[7], x[28]]))
        TCU_Sigint_Table[['Bearing_Var', 'Bearing_Rate_Var']] = Sigint_Plots['CM_Polar'].apply(
            lambda x: pd.Series([x[7], x[28]]))
        a=1

        # TBD : missing firlds
        #
        TCU_Sigint_Table['SPAN_High']=Sigint_Plots['SPAN_High']
        TCU_Sigint_Table['SPAN_Low'] = Sigint_Plots['SPAN_Low']
        TCU_Sigint_Table['Remote_Control'] = Sigint_Plots['Remote_Control']
        TCU_Sigint_Table['id'] = Sigint_Plots['id']
        TCU_Sigint_Table['EPL_Name'] = Sigint_Plots['EPL_Name']
        TCU_Sigint_Table['Frequency_Low'] = Sigint_Plots['Frequency_Low']
        TCU_Sigint_Table['Frequency_High'] = Sigint_Plots['Frequency_High']
        TCU_Sigint_Table['PRI_Low'] = Sigint_Plots['PRI_Low']
        TCU_Sigint_Table['PRI_High'] = Sigint_Plots['PRI_High']
        TCU_Sigint_Table['PW1_Low'] = Sigint_Plots['PW1_Low']
        TCU_Sigint_Table['PW1_High'] = Sigint_Plots['PW1_High']
        TCU_Sigint_Table['Group_ID'] = Sigint_Plots['Group_ID']

        return TCU_Sigint_Table


        # TCU_Sigint_Table.Bearing_CM(:, 2)=0;
        # TCU_Sigint_Table.Bearing_CM(:, 3)=0;
        # N = length(TCU_Sigint_Table.Bearing);
        # TCU_Sigint_Table.Bearing_CM(1: N, 4)=AT_Sys_Params.BearingRateError ^ 2;

    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        return None

def snr_estimation_DAIR(Rsange, loss_mat, Params):
    """
    Estimates the Signal-to-Noise Ratio (SNR) in dB.

    Parameters:
    Rsange (float): Range in meters
    loss_mat (float): Additional losses in dB
    AT_Sys_Params (dict): Dictionary containing system parameters

    Returns:
    float: Estimated SNR in dB
    """
    from my_subscript import getParam
    import math
    try:
        # Read system parameters (if needed)
        Pt_W=getParam(Params,'AT_Sys_Params.Pt_W')

        # Transmitted power in dB
        Pt_tx_db = 10 * math.log10(getParam(Params,'AT_Sys_Params.Pt_W') * getParam(Params,'AT_Sys_Params.Mtx') * getParam(Params,'AT_Sys_Params.Ntx'))
        Gr_db = getParam(Params,'AT_Sys_Params.Gr_db') # Receiver gain in dB
        Gt_db = getParam(Params,'AT_Sys_Params.Gt_db')  # Transmitter gain in dB
        lambda_m = getParam(Params,'AT_Sys_Params.lambda') # Wavelength in meters
        K_jK = getParam(Params,'AT_Sys_Params.k_jK')  # Boltzmann constant * system loss factor
        T_K = getParam(Params,'AT_Sys_Params.T_K')  # System temperature in Kelvin
        loss_db = getParam(Params,'AT_Sys_Params.Loss_db')  # Losses in dB
        noise_figure_db = getParam(Params,'AT_Sys_Params.noise_figure_db')  # Noise figure in dB

        # Coherent processing time
        Tcoh_sec = getParam(Params,'AT_Sys_Params.Tcoh_sec')

        # Pulse repetition interval (PRI)
        pri_sec = 1 / getParam(Params, 'AT_Sys_Params.PRF_hz')


        # Duty cycle calculation
        DC = (getParam(Params, 'AT_Sys_Params.pw_msec')/ 1000) / pri_sec




        # Generate random RCS based on Swerling type
        if getParam(Params, 'AT_Sys_Params.swerling_type') == 1:

            sw_rcs_m2 = getParam(Params, 'AT_Sys_Params.rcs_m2')

        elif getParam(Params, 'AT_Sys_Params.swerling_type') == 3:
            sw_rcs_m2 = getParam(Params, 'AT_Sys_Params.rcs_m2')

        else:
            raise ValueError("Unsupported Swerling type! Only types I and III are allowed.")

        # Full equation components
        snr_numerator = (Pt_tx_db + Gr_db + Gt_db + 2 * 10 * math.log10(lambda_m) +
                         10 * math.log10(sw_rcs_m2) + 10 * math.log10(Tcoh_sec * DC))

        snr_denominator = (3 * 10 * math.log10(4 * math.pi) + 4 * 10 * math.log10(Rsange) +
                           10 * math.log10(K_jK) + 10 * math.log10(T_K) +
                           (loss_db + loss_mat) + noise_figure_db)

        # Compute SNR in dB
        snr_db = snr_numerator - snr_denominator
        return snr_db

    except Exception as err:
        print(f"Error: {err}")
        return None



def Pd_case1(SNR_dB, P_fa):
    import numpy as np
    """
    Computes the probability of detection (Pd) for Swerling Case 1.

    Parameters:
    SNR_dB (list or np.array): Signal-to-noise ratio in dB.
    P_fa (float): Probability of false alarm.

    Returns:
    np.array: Probability of detection (Pd) values.
    """
    SNR_dB = np.atleast_1d(SNR_dB)
    M = len(SNR_dB)

    # Compute detection thresholds in dB
    Vt_dB = 10 * np.log10(-np.log(P_fa))  # dB threshold above (2*sigma_n^2)
    Vt1_dB = Vt_dB + 10 * np.log10(2)  # dB threshold above sigma_n^2
    Vt1 = 10 ** (Vt1_dB / 20)

    # Convert SNR from dB to linear scale
    SNR = 10 ** (np.array(SNR_dB) / 10)

    # Monte Carlo simulation settings
    N_trials = int(1e6)  # Evaluates Pd up to 0.01% accuracy (99.99%)
    sigma_n = 1
    Pd = np.zeros(M)

    for k in range(M):
        sigma_t = sigma_n * np.sqrt(SNR[k])

        # Generate noise and signal + noise matrices
        An_matrix = sigma_n * np.random.randn(N_trials, 2)
        At_matrix = sigma_t * np.random.randn(N_trials, 2)

        # Compute target amplitude and noise+target amplitude
        At = np.sqrt(np.sum(At_matrix ** 2, axis=1))

        As_plus_n = np.sqrt((An_matrix[:, 0] + At) ** 2 + An_matrix[:, 1] ** 2)

        # Compute probability of detection
        Pd[k] = np.sum(As_plus_n > Vt1) / N_trials

    return Pd



def Pd_case3(SNR_dB, P_fa):
    import numpy as np
    """
    Computes the probability of detection (Pd) for Swerling Case 3 (Single Dominant Scatterer).

    Parameters:
    SNR_dB (list or np.array): Signal-to-noise ratio in dB.
    P_fa (float): Probability of false alarm.

    Returns:
    np.array: Probability of detection (Pd) values.
    """
    SNR_dB = np.atleast_1d(SNR_dB)
    M = len(SNR_dB)

    # Compute detection thresholds in dB
    Vt_dB = 10 * np.log10(-np.log(P_fa))  # dB threshold above (2*sigma_n^2)
    Vt1_dB = Vt_dB + 10 * np.log10(2)  # dB threshold above sigma_n^2
    Vt1 = 10**(Vt1_dB / 20)

    # Convert SNR from dB to linear scale
    SNR = 10**(np.array(SNR_dB) / 10)

    # Monte Carlo simulation settings
    N_trials = int(1e6)  # Evaluates Pd up to 0.01% accuracy (99.99%)
    sigma_n = 1
    Pd = np.zeros(M)

    for k in range(M):
        sigma_t = sigma_n * np.sqrt(SNR[k] / 2)

        # Generate noise and signal + noise matrices
        An_matrix = sigma_n * np.random.randn(N_trials, 2)
        At_matrix = sigma_t * np.random.randn(N_trials, 4)  # 4 random components

        # Compute target amplitude and noise+target amplitude
        At = np.sqrt(np.sum(At_matrix**2, axis=1))

        As_plus_n = np.sqrt((An_matrix[:, 0] + At) ** 2 + An_matrix[:, 1] ** 2)

        # Compute probability of detection
        Pd[k] = np.sum(As_plus_n > Vt1) / N_trials

    return Pd







