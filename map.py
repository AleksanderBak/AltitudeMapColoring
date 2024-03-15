import numpy as np
import gradients as grad
import matplotlib.pyplot as plt
import math


def load_map(file_name):
    with open(file_name) as file:
        map = file.read().splitlines()
    map = [i.split(' ') for i in map]
    height = int(map[0][0])
    width = int(map[0][1])
    distance = int(map[0][2])
    map.pop(0)

    for i in range(len(map)):
        map[i].pop(-1)
        map[i] = [float(x) for x in map[i]]

    return map, height, width, distance


def draw_map(map, height, width, distance):
    sun = np.array([distance, 60, distance])
    color_matrix = [[[1, 1, 1] for x in range(height)] for y in range(width)]
    angles = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            a = np.array([i * distance, map[i][j], j * distance])

            if i == 0:
                if j > 0:
                    b = np.array([i * distance, map[i][j - 1], (j - 1) * distance])
                    c = np.array([(i + 1) * distance, map[i + 1][j], j * distance])
                else:
                    b = np.array([i * distance, map[i][j + 1], (j + 1) * distance])
                    c = np.array([(i + 1) * distance, map[i + 1][j], j * distance])
            else:
                if j > 0:
                    b = np.array([i * distance, map[i][j - 1], (j - 1) * distance])
                    c = np.array([(i - 1) * distance, map[i - 1][j], j * distance])
                else:
                    b = np.array([i * distance, map[i][j + 1], (j + 1) * distance])
                    c = np.array([(i - 1) * distance, map[i - 1][j], j * distance])

            vec_to_sun = sun - a
            normal = np.cross(b - a, c - a)  # vector perpendicular to the surface of triangle abc
            vec_to_sun = vec_to_sun / np.linalg.norm(vec_to_sun)
            normal = normal / np.linalg.norm(normal)
            angles[i][j] = math.degrees(np.arccos(np.clip(np.dot(normal, vec_to_sun), -1, 1)))

    sorted_angles = np.sort(angles.flatten())

    min_height = np.min(map)
    max_height = np.max(map) - min_height
    for i in range(height):
        for j in range(width):
            color_matrix[i][j][0] = (1 - ((map[i][j] - min_height) / max_height)) * 120  # Hue between red (0) and green(120)
            position = np.where(sorted_angles == angles[i][j])[0][0]/len(sorted_angles)

            if position - 0.8 > 0:
                color_matrix[i][j][2] -= np.sin(angles[i][j]) * abs(position - 0.3)

            if angles[i][j] < 89.9:
                color_matrix[i][j][1] = ((1 + (angles[i][j] - 90)) + color_matrix[i][j][1]) / 1.95
            else:
                color_matrix[i][j][2] = ((1 - (angles[i][j] - 90)) + color_matrix[i][j][2]) / 1.95

            color_matrix[i][j] = grad.hsv2rgb(color_matrix[i][j][0], color_matrix[i][j][1], color_matrix[i][j][2])
    return color_matrix


if __name__ == '__main__':
    map, height, width, distance = load_map('big.dem')
    mapDrawing = draw_map(map, height, width, distance)
    fig, g1 = plt.subplots()
    g1.tick_params(axis="y", direction="in", left="off", labelleft="on")

    g1.set_xticks([0, 100, 200, 300, 400, 500])
    x_ticks = g1.xaxis.get_major_ticks()
    x_ticks[-1].label1.set_visible(False)

    g1.set_yticks([500, 400, 300, 200, 100, 0])
    y_ticks = g1.yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    g1.invert_yaxis()

    g1_1 = plt.gca().secondary_xaxis('top')
    g1_1.set_xticklabels([])

    g1_2 = plt.gca().secondary_yaxis('right')
    g1_2.tick_params(direction="in")
    g1_2.set_yticklabels([])

    g1.imshow(mapDrawing)
    fig.savefig('map.pdf')
