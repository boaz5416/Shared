def Rad2Deg (rad):
    import math
    return rad*180/math.pi

def Deg2Rad (deg):
    import math
    return deg*math.pi/180

def g_circle(longitude1,latitude1,longitude2,latitude2,num_of_segments):

    import math

    ptlon1 = longitude1
    ptlat1 = latitude1
    ptlon2 = longitude2
    ptlat2 = latitude2

    numberofsegments = num_of_segments
    onelessthansegments = numberofsegments - 1
    fractionalincrement = (1.0/onelessthansegments)

    ptlon1_radians = math.radians(ptlon1)
    ptlat1_radians = math.radians(ptlat1)
    ptlon2_radians = math.radians(ptlon2)
    ptlat2_radians = math.radians(ptlat2)

    distance_radians=2*math.asin(math.sqrt(math.pow((math.sin((ptlat1_radians-ptlat2_radians)/2)),2) + math.cos(ptlat1_radians)*math.cos(ptlat2_radians)*math.pow((math.sin((ptlon1_radians-ptlon2_radians)/2)),2)))
# 6371.009 represents the mean radius of the earth
# shortest path distance
    distance_km = 6371.009 * distance_radians

    mylats = []
    mylons = []

# write the starting coordinates
    mylats.append([])
    mylons.append([])
    mylats[0] = ptlat1
    mylons[0] = ptlon1

    f = fractionalincrement
    icounter = 1
    while (icounter <  onelessthansegments):
        icountmin1 = icounter - 1
        mylats.append([])
        mylons.append([])
        # f is expressed as a fraction along the route from point 1 to point 2
        A=math.sin((1-f)*distance_radians)/math.sin(distance_radians)
        B=math.sin(f*distance_radians)/math.sin(distance_radians)
        x = A*math.cos(ptlat1_radians)*math.cos(ptlon1_radians) + B*math.cos(ptlat2_radians)*math.cos(ptlon2_radians)
        y = A*math.cos(ptlat1_radians)*math.sin(ptlon1_radians) +  B*math.cos(ptlat2_radians)*math.sin(ptlon2_radians)
        z = A*math.sin(ptlat1_radians) + B*math.sin(ptlat2_radians)
        newlat=math.atan2(z,math.sqrt(math.pow(x,2)+math.pow(y,2)))
        newlon=math.atan2(y,x)
        newlat_degrees = math.degrees(newlat)
        newlon_degrees = math.degrees(newlon)
        mylats[icounter] = newlat_degrees
        mylons[icounter] = newlon_degrees
        icounter += 1
        f = f + fractionalincrement

# write the ending coordinates
    mylats.append([])
    mylons.append([])
    mylats[onelessthansegments] = ptlat2
    mylons[onelessthansegments] = ptlon2


# Now, the array mylats[] and mylons[] have the coordinate pairs for intermediate points along the geodesic
# My mylat[0],mylat[0] and mylat[num_of_segments-1],mylat[num_of_segments-1] are the geodesic end points

# write a kml of the results
    zipcounter = 0
    kmlheader = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document><name>LineString.kml</name><open>1</open><Placemark><name>unextruded</name><LineString><extrude>1</extrude><tessellate>1</tessellate><coordinates>"
    print (kmlheader)
    while (zipcounter < numberofsegments):
        klmbody = repr(mylons[zipcounter]) + "," + repr(mylats[zipcounter]) + ",0 "
        print (klmbody)
        zipcounter += 1
    kmlfooter = "</coordinates></LineString></Placemark></Document></kml>"
    print (kmlfooter)
    return mylats,mylons,kmlheader,klmbody,kmlfooter


import math


def calc_bearing(lat1, long1, lat2, long2):
    # Convert latitude and longitude to radians
    lat1 = math.radians(lat1)
    long1 = math.radians(long1)
    lat2 = math.radians(lat2)
    long2 = math.radians(long2)

    # Calculate the bearing
    bearing = math.atan2(
        math.sin(long2 - long1) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(long2 - long1)
    )

    # Convert the bearing to degrees
    bearing = math.degrees(bearing)

    # Make sure the bearing is positive
    bearing = (bearing + 360) % 360

    return bearing

def calc_distance(lat1, long1, lat2, long2):

    from math import sin, cos, sqrt, atan2, radians

# Approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    long1 = radians(long1)
    lat2 = radians(lat2)
    long2 = radians(long2)

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))


    distance = R * c
    return distance

def BuildParams():
    try:
        print ('Running Build Params')
        import pandas as pd
        Params = pd.DataFrame({'Name': [],'Value': []})
        Names=list()
        Values=list()
        for line in open('C:\\Drones\\Params\\'+'Params.txt'):

            try:

                NameValue = line.split('=')
                Name=NameValue[0]


                TempValue=NameValue[1]
                TempValue1=TempValue.split(';')
                TempValue2=TempValue1[0]

                try:
                    Value = float(TempValue2)
                except:
                    Value = TempValue2

                Names.append(Name)
                Values.append(Value)
            except:
                line

        Params = {'Name': Names, 'Value': Values}
        return (Params)
    except:
        print ('xxx')

def BuildEPL():
    print('Running EPL')
    import pandas as pd

    import json


    # Define file path
    fname = r'C:\Drones\Params\EPL.txt'

    # Read JSON file
    with open(fname, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Convert to DataFrame
    EPL = pd.DataFrame(json_data["epls"])
    return EPL

    # Display DataFrame
    # print(df)


def getParam(Params,ParName):
    ind=Params['Name'].index(ParName)
    Values=Params['Value']
    return(Values[ind])




def ECEF2Local_NWU(ecef,lat,lon,hsm,a, e2,v_par):


    import math
    import numpy as np
    import pymap3d as pm
    # Converted from matlab
    np.set_printoptions(precision=16, suppress=True)

    x, y, z = pm.geodetic2ecef(lat, lon, hsm)  #  in deg
    pos0=(x,y,z)

    slat = math.sin(math.radians(lat))
    clat = math.cos(math.radians(lat))
    slon = math.sin(math.radians(lon))
    clon = math.cos(math.radians(lon))

    r11 = -slat * clon
    r12 = slon
    r13 = clat * clon

    r21 = -slat * slon
    r22 = -clon
    r23 = clat * slon

    r31 = clat
    r32 = 0
    r33 = slat

    x = np.array([[r11,r12,r13], [r21,r22,r23],[r31,r32,r33]])
    tge = np.asmatrix(x)

    # tge=np.array(np.asmatrix('r11 r12 r13; r21 r22 r23;r31 r32 r33'), subok=True)

    # TGE = [r11  r12  r13
    #  r21  r22  r239
    #  r31  r32  r33]

    if v_par == 0:
        #Local = TGE*(ecef-pos0)+

        Local=tge.T * (np.array(ecef) - np.array(pos0)).reshape(3, 1)


    else:
        Local = tge.T * (np.array(ecef)).reshape(3, 1)



    return list(Local.flat)


def ECEF2Local_NWU_CovMat(CovMatECEF, lat, lon):
    import math
    import numpy as np
    import pymap3d as pm
    # Converted from matlab





    slat = math.sin(math.radians(lat))
    clat = math.cos(math.radians(lat))
    slon = math.sin(math.radians(lon))
    clon = math.cos(math.radians(lon))

    r11 = -slat * clon
    r12 = slon
    r13 = clat * clon

    r21 = -slat * slon
    r22 = -clon
    r23 = clat * slon

    r31 = clat
    r32 = 0
    r33 = slat


    x = np.array(
        [[r11, r12, r13, 0, 0, 0], [r21, r22, r23, 0, 0, 0], [r31, r32, r33, 0, 0, 0], [0, 0, 0, r11, r12, r13],
         [0, 0, 0, r21, r22, r23], [0, 0, 0, r31, r32, r33]])
    tge = np.asmatrix(x)

    x = np.matmul(np.matmul(tge.T, CovMatECEF), tge)  # tbd *tge1



    return (x)



def Local_NWU2ECEF(Local, lat, lon, hsm, a, e2, v_par):
    import math
    import numpy as np
    import pymap3d as pm
    import traceback
    import sys
    import inspect


    # [Pos0(1,:), Pos0(2,:), Pos0(3,:)] = LLA2ECEF(lat, lon, hsm, a, e2);
    x, y, z = pm.geodetic2ecef(lat, lon, hsm)  # in deg
    pos0 = (x, y, z)



    slat = math.sin(math.radians(lat))
    clat = math.cos(math.radians(lat))
    slon = math.sin(math.radians(lon))
    clon = math.cos(math.radians(lon))

    r11 = -slat * clon
    r12 = slon
    r13 = clat * clon
    r21 = -slat * slon
    r22 = -clon
    r23 = clat * slon
    r31 = clat
    r32 = 0
    r33 = slat

    x = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    tge = np.asmatrix(x)



    if v_par == 0:
        x=np.matmul(tge, np.array(Local)) + np.array(pos0).T



        return(x)

    else:
        x = np.matmul(tge, np.array(Local))
        return (x)

def Local_NWU2ECEF_CovMat(CovMatLocal, lat, lon):
    import math
    import numpy as np
    import pymap3d as pm



    slat = math.sin(math.radians(lat))
    clat = math.cos(math.radians(lat))
    slon = math.sin(math.radians(lon))
    clon = math.cos(math.radians(lon))

    r11 = -slat * clon
    r12 = slon
    r13 = clat * clon
    r21 = -slat * slon
    r22 = -clon
    r23 = clat * slon
    r31 = clat
    r32 = 0
    r33 = slat



    x = np.array([[r11, r12, r13,0,0,0], [r21, r22, r23,0,0,0], [r31, r32, r33,0,0,0],[0,0,0,r11,r12,r13],[0,0,0,r21,r22,r23],[0,0,0,r31,r32,r33]])
    tge = np.asmatrix(x)


    x=np.matmul(np.matmul(tge,CovMatLocal),tge.T)# tbd *tge1

    return (x)






def nwu2enu(nwu_in):
    import numpy as np
    # Convert input list to a NumPy array

    # Get the shape of the input
    try:
        temp=nwu_in.shape
        dim1=temp[0]
        dim2=temp[1]
    except:
        temp = nwu_in.shape
        dim1=temp[0]
        dim2=1




    # Define the rotation matrices
    R1 = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])

    R2 = np.block([[R1, np.zeros((3, 3))],
                   [np.zeros((3, 3)), R1]])

    if dim1 == 6 and dim2 == 6:  # CM
        enu_out = R2 @ nwu_in @ R2.T

    elif dim1 == 3 and dim2 == 3:  # CM
        enu_out = R1 @ nwu_in @ R1.T

    elif dim1 == 1 and dim2 == 3:
        enu_out = (R1 @ nwu_in.T).T

    elif dim1 == 3 and dim2 == 1:
        enu_out = R1 @ nwu_in

    elif dim1 == 1 and dim2 == 6:
        enu_out = (R2 @ nwu_in.T).T

    elif dim1 == 6 and dim2 == 1:
        enu_out = R2 @ nwu_in

    else:
        print("dim of SV is incorrect")
        enu_out = np.array([-1, -1, -1])

    # Convert the output back to a Python list and return it
    return enu_out.tolist()


def enu2spherical_rdot(PosLocal, VelLocal):
    import numpy as np
    """
    Performs coordinate transform from ENU Local Radar to Spherical Radar coordinate system.

    Parameters:
    PosLocal (list): Local coordinates [x, y, z]
    VelLocal (list): Local velocity

    Returns:
    Range (float): Range to radar, meters
    Azimuth (float): Azimuth, rad (from north to east)
    Elevation (float): Elevation, rad
    Rdot (float): m/sec
    """

    # Convert lists to NumPy arrays for easier manipulation
    PosLocal = np.array(PosLocal)
    VelLocal = np.array(VelLocal)

    # Range (distance to the radar)
    Range = np.sqrt(np.sum(PosLocal ** 2))

    # Azimuth (angle from north to east)
    Azimuth = np.arctan2(PosLocal[0], PosLocal[1])

    # Rxy (distance in the xy-plane)
    Rxy = np.sqrt(np.sum(PosLocal[0:2] ** 2))

    # Elevation (angle from horizontal plane)
    Elevation = np.arctan2(PosLocal[2], Rxy)

    # Rdot (rate of change of position alon the direction of PosLocal)
    Rdot = np.dot(VelLocal, PosLocal) / Range

    return Range, Azimuth, Elevation, Rdot


def Calc_dAlpha(Alpha1, Alpha2):

    import math
    s = math.sin(Alpha1 - Alpha2)
    c = math.cos(Alpha1 - Alpha2)
    dAlpha = math.atan2(s, c)
    return dAlpha

def Shere2Cart(Measurements):
    import numpy as np
    try:
        # Convert polar to cartesian coordinates
        Measurements['x'] = Measurements['Range'] * np.cos(Measurements['Az']) * np.cos(Measurements['El'])
        Measurements['y'] = -Measurements['Range'] * np.sin(Measurements['Az']) * np.cos(Measurements['El'])
        Measurements['z'] = Measurements['Range'] * np.sin(Measurements['El'])

        # Initialize the Jacobian matrix H
        H = np.zeros((3, 3))

        H[0, 0] = np.cos(float(Measurements['Az'].iloc[0])) * np.cos(float(Measurements['El'].iloc[0]))  # DX/DR
        H[0, 1] = -float(Measurements['Range'].iloc[0]) * np.cos(float(Measurements['El'].iloc[0])) * np.sin(
            float(Measurements['Az'].iloc[0]))  # DX/DAZ
        H[0, 2] = -float(Measurements['Range'].iloc[0]) * np.cos(float(Measurements['Az'].iloc[0])) * np.sin(
            float(Measurements['El'].iloc[0]))  # DX/DEL

        H[1, 0] = -np.cos(float(Measurements['El'].iloc[0])) * np.sin(float(Measurements['Az'].iloc[0]))  # DY/DR
        H[1, 1] = -float(Measurements['Range'].iloc[0]) * np.cos(float(Measurements['Az'].iloc[0])) * np.cos(
            float(Measurements['El'].iloc[0]))  # DY/DAZ
        H[1, 2] = float(Measurements['Range'].iloc[0]) * np.sin(float(Measurements['Az'].iloc[0])) * np.sin(
            float(Measurements['El'].iloc[0]))  # DY/DEL

        H[2, 0] = np.sin(float(Measurements['El'].iloc[0]))  # DZ/DR

        # Covariance matrix in polar coordinates
        CM_Polar = np.zeros((3, 3))
        CM_Polar[0, 0] = float(Measurements['RangeError'].iloc[0]) ** 2
        CM_Polar[1, 1] = float(Measurements['AzError'].iloc[0]) ** 2
        CM_Polar[2, 2] = float(Measurements['ElError'].iloc[0]) ** 2

        # Perform element-wise multiplication
        result = (H * CM_Polar) * H.T  # Element-wise multiplication, then multiply by transpose

        # Flatten the result
        reshaped_result = result.flatten()  # Convert the matrix to a 1D array with 9 elements

        # Store the result as a list of 9 items in the single row of the DataFrame
        Measurements['CM_Cart'] = [reshaped_result.tolist()]  # Assign as a list of lists


        a=1







        return Measurements, H

    except Exception as err:
        print("Error:", err)
        return None, None


def ConvertSphere2Cartesian(SV_Sphere, CM_Sphere=None):
    try:

        import numpy as np
        import traceback
        from scipy.linalg import cholesky
        from my_subscript import Sphere_To_Cart,GaussianApprox
        if CM_Sphere is None:  # Only SV
            SV_Cart = Sphere_To_Cart(SV_Sphere)
            CM_Cart = 0
        else:
            if CM_Sphere.shape[1] == 36:  # CM is given as vector of 36
                SV_Cart = Sphere_To_Cart(SV_Sphere)
                CM_Cart = np.zeros_like(CM_Sphere)
                for p in range(CM_Sphere.shape[0]):
                    CM_Temp = CM_Sphere[p, :].reshape(6, 6)
                    L = cholesky(6 * CM_Temp, lower=True)
                    Sphere_Points = np.vstack([(-L + SV_Sphere[p, :]).T, (L + SV_Sphere[p, :]).T]).T
                    Cart_Points = Sphere_To_Cart(Sphere_Points)
                    _, CM_Cart_Temp = GaussianApprox(Cart_Points, True)
                    CM_Cart[p, :] = CM_Cart_Temp.flatten()
            else:
                if SV_Sphere.shape[0] == 6:  # If column, make it a row
                    SV_Sphere = SV_Sphere.T
                SV_Cart = Sphere_To_Cart(SV_Sphere)
                L = cholesky(6 * CM_Sphere, lower=True)
                Sphere_Points = np.vstack([(-L + SV_Sphere).T, (L + SV_Sphere).T]).T
                Cart_Points = Sphere_To_Cart(Sphere_Points)
                _, CM_Cart = GaussianApprox(Cart_Points, True)

        return SV_Cart, CM_Cart

    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        return None



def Sphere_To_Cart(SV_Sphere):

    from my_subscript import POLAR2LGC

    Range = SV_Sphere[0]
    Az = SV_Sphere[1]
    El = SV_Sphere[2]
    Rdot = SV_Sphere[3]
    Azdot = SV_Sphere[4]
    Eldot = SV_Sphere[5]

    x = Range * np.cos(Az) * np.cos(El)
    y = -Range * np.sin(Az) * np.cos(El)
    z = Range * np.sin(El)


    vx = Rdot * np.cos(El) * np.cos(Az) - Range * Azdot * np.cos(El) * np.sin(Az) - Range * Eldot * np.sin(El) * np.cos(Az)
    vy = -Rdot * np.cos(El) * np.sin(Az) - Range * Azdot * np.cos(El) * np.cos(Az) + Range * Eldot * np.sin(El) * np.sin(Az)
    vz = Rdot * np.sin(El) + Range * Eldot * np.cos(El)

    SV_Cart = np.vstack([x, y, z, vx, vy, vz]).T
    return SV_Cart





def GaussianApprox(Points, Unscented):
    if Points.shape[1] > Points.shape[0]:
        Points = Points.T

    N, _ = Points.shape
    SV_Out = np.mean(Points, axis=0)

    if Unscented:
        M = N
    else:
        M = N - 1

    beta = Points - SV_Out
    CM_Out = np.dot(beta.T, beta) / M

    return SV_Out, CM_Out



def POLAR2LGC(plr):
    """
    Converts polar coordinates (azimuth, elevation, range) to local Cartesian NWU.

    Parameters:
    plr : numpy.ndarray
        A 2D array of shape (3, N), where:
        - plr[0, :] is azimuth in radians
        - plr[1, :] is elevation in radians
        - plr[2, :] is range in meters

    Returns:
    numpy.ndarray
        A 2D array of shape (3, N), representing (n, w, u) coordinates in meters.
    """

    import numpy as np
    r = plr[2]
    u = r * np.sin(plr[1])
    n = r * np.cos(plr[1]) * np.cos(plr[0])
    w = r * np.cos(plr[1]) * np.sin(plr[0])

    nwu = np.vstack([n, w, u])
    return nwu

import numpy as np

def Predict_Track(SV_Prev, CM_Prev, dT, alpha, SigmaM):
    import traceback
    """
    Predicts track SV and CM.

    Inputs:
    SV_Prev: (9,1) array - [x,y,z,vx,vy,vz,ax,ay,az]
    CM_Prev: (9,9) array - same order as SV_Prev
    dT: time period from Prev to Pred
    alpha: motion model (0 = acceleration is white noise, inf = constant acceleration)
    SigmaM: Average acceleration in m/sec^2

    Outputs:
    SV_Pred: predicted state vector
    CM_Pred: predicted covariance matrix
    """
    try:
        lamdba = 1
        alphah = alpha
        SigmaMh = SigmaM

        Proj_Matrix1 = np.zeros((9, 9))
        Proj_Matrix1[0, 0] = 1
        Proj_Matrix1[1, 3] = 1
        Proj_Matrix1[2, 6] = 1
        Proj_Matrix1[3, 1] = 1
        Proj_Matrix1[4, 4] = 1
        Proj_Matrix1[5, 7] = 1
        Proj_Matrix1[6, 2] = 1
        Proj_Matrix1[7, 5] = 1
        Proj_Matrix1[8, 8] = 1

        Proj_Matrix = Proj_Matrix1.T  # Transpose since Proj_Matrix1 is symmetrical

        State = Proj_Matrix @ SV_Prev  # Convert state representation
        CM = Proj_Matrix @ CM_Prev @ Proj_Matrix.T

        q = np.zeros((3, 3))
        At = alpha * dT

        q[0, 0] = (0.5 / (alpha**5)) * (1 - np.exp(-2 * At) + 2 * At + (2/3) * (At**3) - 2 * (At**2) - 4 * At * np.exp(-At))
        q[0, 1] = (0.5 / (alpha**4)) * (np.exp(-2 * At) + 1 - 2 * np.exp(-At) + 2 * At * np.exp(-At) - 2 * At + At**2)
        q[0, 2] = (0.5 / (alpha**3)) * (1 - np.exp(-2 * At) - 2 * At * np.exp(-At))
        q[1, 1] = (0.5 / (alpha**3)) * (4 * np.exp(-At) - 3 - np.exp(-2 * At) + 2 * At)
        q[1, 2] = (0.5 / (alpha**2)) * (np.exp(-2 * At) + 1 - 2 * np.exp(-At))
        q[2, 2] = (0.5 / alpha) * (1 - np.exp(-2 * At))

        q[1, 0] = q[0, 1]
        q[2, 0] = q[0, 2]
        q[2, 1] = q[1, 2]

        qh = q  # For IMM algorithm only

        SigmaM2 = SigmaM**2
        Q = np.zeros((9, 9))
        Q[:3, :3] = 2 * alpha * SigmaM2 * q
        Q[3:6, 3:6] = 2 * alpha * SigmaM2 * q
        Q[6:9, 6:9] = 2 * alpha * SigmaM2 * q  # TBD: Check if Z components in Q are 0

        f13 = (1 / alpha**2) * (-1 + At + np.exp(-At))
        f23 = (1 / alpha) * (1 - np.exp(-At))
        f33 = np.exp(-At)

        F = np.zeros((9, 9))
        f = np.zeros((3, 3))

        f[0, 0] = 1
        f[0, 1] = dT
        f[0, 2] = f13
        f[1, 1] = 1
        f[1, 2] = f23
        f[2, 2] = f33

        fh = np.zeros((3, 3))
        fh[0, 0] = 1
        fh[0, 1] = dT
        fh[0, 2] = f13
        fh[1, 1] = 1
        fh[1, 2] = f23
        fh[2, 2] = f33

        F[:3, :3] = f
        F[3:6, 3:6] = f
        F[6:9, 6:9] = fh

        SV_Pred1 = F @ State
        SV_Pred = Proj_Matrix1 @ SV_Pred1

        lambda_pr = 1
        CM_Pred1 = lambda_pr * (F @ CM @ F.T) + Q
        CM_Pred = Proj_Matrix1 @ CM_Pred1 @ Proj_Matrix1.T

        return SV_Pred, CM_Pred

    except Exception as err:
        print("Error occurred:")
        traceback.print_exc()  # Prints the full stack trace
        return None, None





def ConvertCartesian2Sphere(SV_Cart, CM_Cart=None):
    import numpy as np
    import traceback

    """
    Converts a state vector and covariance matrix from Cartesian to Spherical coordinates.

    Args:
        SV_Cart (np.ndarray): State vector in Cartesian coordinates [x, y, z, vx, vy, vz] (NWU frame).
        CM_Cart (np.ndarray, optional): Covariance matrix in Cartesian coordinates.

    Returns:
        tuple: (SV_Sphere, CM_Sphere, H)
    """
    try:

        # SV_Cart = np.asarray(SV_Cart)
        SV_Cart=SV_Cart.reshape(1, 6)
        SV_Sphere,H = Cart_To_Sphere(SV_Cart,1)

        if CM_Cart is None:
            SV_Sphere,dummy = Cart_To_Sphere(SV_Cart,0)
            CM_Sphere = 0
        else:
            CM_Cart = np.asarray(CM_Cart)

            if CM_Cart.shape[1] == 36:
                SV_Sphere,dummy = Cart_To_Sphere(SV_Cart,0)
                CM_Sphere = np.zeros_like(CM_Cart)

                for p in range(CM_Cart.shape[0]):
                    CM_Temp = CM_Cart[p, :].reshape(6, 6).T
                    L = np.linalg.cholesky(6 * CM_Temp)
                    Cart_Points = np.vstack([-L, L]) + SV_Cart[p, :]
                    Sphere_Points,dummy = Cart_To_Sphere(Cart_Points,0)
                    _, CM_Sphere_Temp = GaussianApprox(Sphere_Points, True)
                    CM_Sphere[p, :] = CM_Sphere_Temp.ravel()
            else:

                SV_Sphere,dummy = Cart_To_Sphere(SV_Cart,0)
                L = np.linalg.cholesky(6 * CM_Cart)

                Cart_Points = (np.vstack([-L, L]).T + np.tile(SV_Cart.T, (1, 12))).T
                Sphere_Points,dummy = Cart_To_Sphere(Cart_Points,0)
                _, CM_Sphere = GaussianApprox(Sphere_Points, True)

        return SV_Sphere, CM_Sphere, H
    except Exception as err:
            print("An error occurred:")
            traceback.print_exc()
            a=1# This will print the full stack trace



def Cart_To_Sphere(Cart_Points, Calculate_H):
    import numpy as np
    import traceback
    try:
        x = Cart_Points[:,0]
        y = Cart_Points[:,1]
        z = Cart_Points[:,2]

        vx = Cart_Points[:,3]
        vy = Cart_Points[:,4]
        vz = Cart_Points[:,5]

        P = np.sqrt(x**2 + y**2)
        Range = np.sqrt(x**2 + y**2 + z**2)
        Az = np.arctan2(-y, x)
        El = np.arctan2(z, P)

        R_dot = (x * vx + y * vy + z * vz) / Range
        Az_dot = (y * vx) / P**2 - (x * vy) / P**2
        El_dot = (x * (x * vz - z * vx) + y * (y * vz - z * vy)) / (P * Range**2)
        Sphere_Points = np.column_stack((Range, Az, El, R_dot, Az_dot, El_dot))

        # Jakobian
        H = np.zeros((4, 6))
        if Calculate_H:

            H[0, 0] = x / Range
            H[0, 1] = y / Range
            H[0, 2] = z / Range

            H[1, 0] = y / P**2
            H[1, 1] = -x / P**2

            H[2, 0] = (-x * z) / (Range**2 * P)
            H[2, 1] = (-y * z) / (Range**2 * P)
            H[2, 2] = P / Range**2

            H[3, 0] = (vx * (y**2 + z**2) - x * (y * vy + z * vz)) / Range**3
            H[3, 1] = (vy * (x**2 + z**2) - y * (x * vx + z * vz)) / Range**3
            H[3, 2] = (vz * (x**2 + y**2) - z * (x * vx + y * vy)) / Range**3
            H[3, 3] = H[0, 0]
            H[3, 4] = H[0, 1]
            H[3, 5] = H[0, 2]

            # symbolic [they are the same !!!!]
            HH = np.zeros((4, 6))

            HH[0, 0] = x / np.sqrt(x**2 + y**2 + z**2)  # dr/dx
            HH[0, 1] = y / np.sqrt(x**2 + y**2 + z**2)  # dr/dy
            HH[0, 2] = z / np.sqrt(x**2 + y**2 + z**2)  # dr/dz
            HH[0, 3] = 0  # dr/dvx
            HH[0, 4] = 0  # dr/dvy
            HH[0, 5] = 0  # dr/dvz

            HH[1, 0] = y / (x**2 + y**2)  # daz/dx
            HH[1, 1] = -x / (x**2 + y**2)  # daz/dy
            HH[1, 2] = 0  # daz/dz
            HH[1, 3] = 0  # daz/dvx
            HH[1, 4] = 0  # daz/dvy
            HH[1, 5] = 0  # daz/dvz

            HH[2, 0] = -(x * z) / (np.sqrt(x**2 + y**2) * (x**2 + y**2 + z**2))  # del/dx
            HH[2, 1] = -(y * z) / (np.sqrt(x**2 + y**2) * (x**2 + y**2 + z**2))  # del/dy
            HH[2, 2] = np.sqrt(x**2 + y**2) / (x**2 + y**2 + z**2)  # del/dz
            HH[2, 3] = 0  # del/dvx
            HH[2, 4] = 0  # del/dvy
            HH[2, 5] = 0  # del/dvz

            HH[3, 0] = vx / np.sqrt(x**2 + y**2 + z**2) - (x * (vx * x + vy * y + vz * z)) / (x**2 + y**2 + z**2)**(3 / 2)  # ddop/dx
            HH[3, 1] = vy / np.sqrt(x**2 + y**2 + z**2) - (y * (vx * x + vy * y + vz * z)) / (x**2 + y**2 + z**2)**(3 / 2)  # ddop/dy
            HH[3, 2] = vz / np.sqrt(x**2 + y**2 + z**2) - (z * (vx * x + vy * y + vz * z)) / (x**2 + y**2 + z**2)**(3 / 2)  # ddop/dz
            HH[3, 3] = x / np.sqrt(x**2 + y**2 + z**2)  # ddop/dvx
            HH[3, 4] = y / np.sqrt(x**2 + y**2 + z**2)  # ddop/dvy
            HH[3, 5] = z / np.sqrt(x**2 + y**2 + z**2)  # ddop/dvz
        S = 1

        return Sphere_Points, H
    except Exception as err:
        print("An error occurred:")
        traceback.print_exc()  # This will print the full stack trace



def GaussianApprox(Points, Unscented):
    import numpy as np
    # Ensure Points is a column-major format (if more columns than rows, transpose)
    if Points.shape[1] > Points.shape[0]:
        Points = Points.T

    N = Points.shape[0]  # Number of rows
    SV_Out = np.mean(Points, axis=0)  # Compute mean along columns (1x6 vector)

    M = N if Unscented else N - 1  # Choose denominator based on Unscented flag

    beta = Points - SV_Out  # Subtract mean from each row (12x6)
    CM_Out = (beta.T @ beta) / M  # Compute covariance-like matrix (6x6)

    return SV_Out, CM_Out



def load_figure(file_path):
    import pickle
    with open(file_path.strip("'"), "rb") as f:  # Strip quotes if needed
        fig = pickle.load(f)
    fig.show()

def my_cdfplot(data, num_points):
    X = np.linspace(min(data), max(data), num_points)
    Y = np.array([np.sum(data < x) / len(data) for x in X])
    return X, Y



def cartesian_to_polar_covariance(cart_covariance, position, velocity):
    """
    Converts Cartesian covariance (position + velocity) to polar (range, azimuth, elevation, Doppler).

    Args:
        cart_covariance: 6x6 covariance matrix (for [x, y, z, vx, vy, vz])
        position: 3D position (x, y, z)
        velocity: 3D velocity (vx, vy, vz)

    Returns:
        4x4 polar covariance matrix (range, azimuth, elevation, Doppler), and the Jacobian
    """
    x, y, z = position
    vx, vy, vz = velocity

    r = np.sqrt(x**2 + y**2 + z**2)
    rho_sq = x**2 + y**2
    rho = np.sqrt(rho_sq)

    if r == 0 or rho_sq == 0:
        raise ValueError("Cannot compute Jacobian at origin or vertical line.")

    # Unit line-of-sight vector
    los = np.array([x, y, z]) / r

    # Doppler = dot(velocity, los)
    # Jacobian for Doppler wrt position = 0
    # Jacobian for Doppler wrt velocity = los
    H = np.zeros((4, 6))

    # Range
    H[0, 0:3] = [x / r, y / r, z / r]

    # Azimuth
    H[1, 0:3] = [-y / rho_sq, x / rho_sq, 0]

    # Elevation
    H[2, 0:3] = [
        x * z / (r**2 * rho),
        y * z / (r**2 * rho),
        rho / r**2
    ]

    # Doppler (radial velocity)
    H[3, 3:6] = los  # partial derivative wrt vx, vy, vz

    # Polar covariance
    polar_cov = H @ cart_covariance @ H.T

    return polar_cov, H


import numpy as np

def cartesian_to_polar(cartesian_6d):
    """
    Converts a 6D Cartesian state [x, y, z, vx, vy, vz] to polar coordinates with Doppler.

    Args:
        cartesian_6d: np.array of shape (6,), with [x, y, z, vx, vy, vz]

    Returns:
        np.array of shape (4,), with [range, azimuth, elevation, doppler]
    """
    x, y, z, vx, vy, vz = cartesian_6d

    # Position vector
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)

    # Line-of-sight unit vector
    los = np.array([x, y, z]) / r

    # Velocity vector
    velocity = np.array([vx, vy, vz])

    # Doppler = projection of velocity onto LOS
    doppler = np.dot(velocity, los)

    return np.array([r, azimuth, elevation, doppler])
















