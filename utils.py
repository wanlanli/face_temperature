import numpy as np


def get_key_points(landmard):
    left_point = np.array(landmard['left_eye'])[
        np.argmax(np.array(landmard['left_eye']),axis=0)[0]]
    right_point = np.array(landmard['right_eye'])[
        np.argmin(np.array(landmard['right_eye']),axis=0)[0]]
    noise_down = np.array(landmard['nose_tip'])[
        np.argmax(np.array(landmard['nose_tip']),axis=0)[1]]
    noise_left_down = np.array(landmard['nose_tip'])[
        np.argmin(np.array(landmard['nose_tip']),axis=0)[0]]
    noise_right_down = np.array(landmard['nose_tip'])[
        np.argmax(np.array(landmard['nose_tip']),axis=0)[0]]
    points = np.array([left_point,right_point,noise_right_down,noise_down,noise_left_down,left_point])
    name = ["left_point","right_point","noise_right_down","noise_down","noise_left_down","left_point"]
    return points, name

def _get_value(mask):
    flatten = mask.flatten()
    value = flatten[flatten>0]
    return value
def describe_temperature(mask):
    value = _get_value(mask)
    return np.mean(value), np.max(value), np.min(value)

