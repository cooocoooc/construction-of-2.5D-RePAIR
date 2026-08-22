""" color_utils.py for color computing method """

import numpy as np 

def _keep_shape(input_value, result_arr):
    """ 
    input type: int, float, etc. 
    output type: the type of result_arr convert to scalar
    input type: array, list, etc.
    output type: result_arr -> no convert
    """

    if np.isscalar(input_value):
        return result_arr.item()
    return result_arr
def _check_empty(arr: np.ndarray) -> bool:
    """ check if the array has element"""
    return arr.size == 0

def scale_convert(h_value, input_scale: str = 'opencv'):
    """ 180 <-> 360 degree exchange

    Args:
        h_value: 
               - type: scalar or array
        input_scale: 'opencv' or 'full'
    
    Return:
        the type of converted value: the scalar or array
   
    """

    h_arr = h_value

    if h_arr.size == 0:
        return h_arr

    if input_scale == 'opencv':
        result = h_arr * 2.0
    elif input_scale == 'full':
        result = h_arr / 2.0
    else:
        return h_value

    return _keep_shape(h_value, result)

def wrap_angle(angle_value):
    """ angle normalized to the range[0, 360]
    
    Args:
        angle_value: degree
            - type: scalar or array
    Return: normalized degree
        - type: scalar or array
    """
    angle_arr = np.asarray(angle_value, dtype = np.float64) % 360.0
    return _keep_shape(angle_value, angle_arr)

def radian_convert(angle_value):
    """ angle degree to radian

    Args:
        angle_value: degree
            - type: scalar or array
    Return: radian
            - type: scalar or array
    """

    rad_arr = np.radians(angle_value)
    return _keep_shape(angle_value, rad_arr)

def degree_convert(rad_value):
    """ radian to angle degree

    Args:
        rad_value: radian
            - type: scalar or array
    Return: degree
            - type: scalar or array
    """

    angle_arr = np.degrees(rad_value) % 360.0
    return _keep_shape(rad_value, angle_arr)       

def circular_mean(h_values, scale: str = 'opencv', output_scale: str = 'opencv') -> float:
    """ computing the mean of the hue based on the hue ring (0-360 degree)

    Args:
        h_value: the degree distribution
        input_scale: 'opencv' or 'full'
        output_scale: 'opencv' or 'full'
    Return: the mean of the hue ring
    """

    real_deg = scale_convert(h_values, scale) 
    if _check_empty(real_deg):
        return np.nan
    
    rads = radian_convert(real_deg)
    mean_x = np.mean(np.cos(rads))
    mean_y = np.mean(np.sin(rads))
    mean_deg = degree_convert(np.arctan2(mean_y, mean_x)) 

    if output_scale == 'opencv':
        return mean_deg / 2.0
    
    return mean_deg


    
