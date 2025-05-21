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
import glob
import shutil




def Sim(Scenario,Build_White_Picture,Build_Plots,Build_Tracks,BuildVisualization,ShowTracksErrors,ShowPlotsErrors, debug=True):

    if False:
        import pdb;pdb.set_trace()
    if __name__ == "__main__" or sys.modules.get("cython") is None:
        from Drones_Not_Compiled import DefineScenario,BuildScenario,Visualization,Show_Tracks_Errors,Show_Plots_Errors,ImportScenario
    from Drones_Compiled import BuildDairPlots, Drones_Tracker,  Export_Tracks_To_TCU, Export_Sigint_To_TCU

    import pandas as pd
    import tables
    import os
    import numpy as np
    from scipy.io import loadmat
    import scipy.io



    ScenarioFolder = "C:\\Drones\\Python\\Scenarios\\" + Scenario + "\\"



    Params=BuildParams()

    # Convert to DataFrame (one row, keys as columns)
    # df = pd.DataFrame([Params])
    df = pd.DataFrame(Params.items(), columns=['Key', 'Value'])

    # Export to Excel


    # Convert to dictionary and reconstruct properly



    #############
    # Extract keys and values
    keys = df.loc[df['Key'] == 'Name', 'Value'].values[0]
    values = df.loc[df['Key'] == 'Value', 'Value'].values[0]

    # Create two-column DataFrame
    df = pd.DataFrame({'Key': keys, 'Value': values})

    # Export to Excel
    df.to_excel(ScenarioFolder+'Params.xlsx', index=False)

    source_path = 'C:\\Drones\\Params\\Params.txt'

    destination_path = ScenarioFolder+'Params.txt'

    shutil.copy(source_path, destination_path)
    ####################


    EPL=BuildEPL()

    Fname = os.path.join(ScenarioFolder, 'SP_Data.h5')
    if not Build_White_Picture == 'External CSV':

        (Sensors,WayPoints)=DefineScenario(ScenarioFolder,Params,0)
    else:
        (Sensors) = DefineScenario(ScenarioFolder, Params, 1)

    if os.path.exists(Fname):
        Sensors.to_hdf(Fname, key='Sensors', mode='a')
    else:
        Sensors.to_hdf(Fname, key='Sensors', mode='w')
    if not Build_White_Picture == 'External CSV':
        WayPoints.to_hdf(Fname, key='WayPoints', mode='a')












    if Build_White_Picture=='Rebuild':

        print ('White_Picture started')
        White_Picture=BuildScenario(WayPoints,Params)
        if os.path.exists(Fname):
            White_Picture.to_hdf(Fname, key='White_Picture', mode='a')
        else:
            White_Picture.to_hdf(Fname, key='White_Picture', mode='w')

        # Beep at 1000 Hz for 500 milliseconds



        print ('White_Picture completed')


    elif Build_White_Picture=='Load Python':
        White_Picture = pd.read_hdf(Fname, key='White_Picture')

    elif Build_White_Picture=='Load Matlab':
        White_Picture = pd.read_excel(ScenarioFolder +'White_Picture.xlsx')

    elif Build_White_Picture == 'External CSV':
        TrajFile = glob.glob(ScenarioFolder+"*Traj*.csv")
        print (TrajFile)
        Traj = pd.read_csv(TrajFile[0])
        White_Picture = ImportScenario(Traj, Params)

        if os.path.exists(Fname):
            White_Picture.to_hdf(Fname, key='White_Picture', mode='a')
        else:
            White_Picture.to_hdf(Fname, key='White_Picture', mode='w')
        pass







    if Build_Plots:

        print ('Plots started')
        # Sensors,WayPoint=DefineScenario(ScenarioFolder,Params)
        Drone_Plots=BuildDairPlots(White_Picture,Sensors,Params,EPL)
        if os.path.exists(Fname):
            Drone_Plots.to_hdf(Fname, key='Drone_Plots', mode='a')
        else:
            Drone_Plots.to_hdf(Fname, key='Drone_Plots', mode='w')
        print ('Plots Completed')

    else:
        Drone_Plots = pd.read_hdf(Fname, key='Drone_Plots')
    All_Drone_Plots=Drone_Plots

    All_Drone_Plots.to_excel(ScenarioFolder + "Drone_Plots.xlsx", sheet_name="Plots", index=False)


    All_Sensors=Sensors

    if Build_Tracks:
        Sensors=All_Sensors[All_Sensors['SensorType']==1]
        Drone_Plots=All_Drone_Plots[All_Drone_Plots['SensorType']==1]
        Drone_Tracks=Drones_Tracker (ScenarioFolder,Params,Sensors,Drone_Plots)
        Drone_Tracks.to_excel(ScenarioFolder + "Drone_Tracks.xlsx", sheet_name="Tracks", index=False)

        if os.path.exists(Fname):

            Drone_Tracks.to_hdf(Fname, key='Drone_Tracks', mode='a')
        else:
            Drone_Tracks.to_hdf(Fname, key='Drone_Tracks', mode='w')



    else:

        Drone_Tracks = pd.read_hdf(Fname, key='Drone_Tracks')

    Drone_Tracks.to_excel(ScenarioFolder + "Drone_Tracks.xlsx", sheet_name="Tracks", index=False)
    if not Build_White_Picture == 'External CSV':
        TCU_Tracks_Table=Export_Tracks_To_TCU(Drone_Tracks, Sensors)

        if os.path.exists(Fname):
          TCU_Tracks_Table.to_hdf(Fname, key='TCU_Tracks_Table', mode='a')
        else:
            TCU_Tracks_Table.to_hdf(Fname, key='TCU_Tracks_Table', mode='w')


        TCU_Tracks_Table.to_excel(ScenarioFolder + "TCU_Tracks_Table.xlsx", sheet_name="Tracks", index=False)

        Sigint_Plots=All_Drone_Plots[All_Drone_Plots['SensorType']==2]
        Sigint_Sensors = All_Sensors[All_Sensors['SensorType'] == 2]
        TCU_Sigint_Table = Export_Sigint_To_TCU(Sigint_Plots, Sigint_Sensors,EPL,Params)

        if os.path.exists(Fname):
          TCU_Sigint_Table.to_hdf(Fname, key='TCU_Sigint_Table', mode='a')
        else:
            TCU_Sigint_Table.to_hdf(Fname, key='TCU_Sigint_Table', mode='w')

        TCU_Sigint_Table.to_excel(ScenarioFolder + "TCU_Sigint_Table.xlsx", sheet_name="Tracks", index=False)



        with pd.HDFStore(Fname, mode='r') as store:
            keys = store.keys()

        print(keys)


    if BuildVisualization:
        Visualization(ScenarioFolder,White_Picture,Drone_Plots,Sensors,Drone_Tracks)

    if ShowTracksErrors:
        Show_Tracks_Errors(ScenarioFolder, Drone_Tracks, Sensors, White_Picture)

    if ShowPlotsErrors:
        Drone_Plots = All_Drone_Plots[All_Drone_Plots['SensorType'] == 1]
        Show_Plots_Errors(ScenarioFolder, Drone_Plots, Sensors, White_Picture,Params)




