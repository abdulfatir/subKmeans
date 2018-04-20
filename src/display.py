from __future__ import print_function
import matplotlib.pyplot as plt
import random
random.seed(1337)


def log_time(message, time, tabs=1):
    message = (tabs * '\t') + '[t] ' + message
    print('%s: %.4fs' % (message, time))


def assign_markers(Y):
    markers = ['o', 'v', '^', '<', '>', '8', 's',
               'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    random.shuffle(markers)
    marker_map = {}
    M = []
    idx = 0
    for y in Y:
        m = marker_map.get(y)
        if m:
            M.append(m)
        else:
            marker_map[y] = markers[idx]
            idx += 1
            M.append(marker_map[y])
    return M


def assign_colors(Y):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {}
    C = []
    idx = 0
    for y in Y:
        m = color_map.get(y)
        if m:
            C.append(m)
        else:
            color_map[y] = colors[idx]
            idx += 1
            C.append(color_map[y])
    return C
