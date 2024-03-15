import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import math

samples = 512
rgb_bw = [[0, 0, 0], [1, 1, 1]]
rgb_gbr = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
rgb_gbr_full = [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]]
rgb_wb_custom = [[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
hsv_bw = [[0, 0, 0], [0, 0, 1]]
hsv_gbr = [[120, 1, 1], [180, 1, 1], [240, 1, 1], [300, 1, 1], [360, 1, 1]]
hsv_unknown = [[119, 0.5, 1], [58, 0.5, 1], [0, 0.53, 0.97]]
hsv_custom = [[10, 0.19, 0.92], [160, 0.6, 0.91], [296, 0.24, 0.23]]

plt.rcParams['xtick.direction'] = 'in'
plt.xmargin = 20
plt.ymargin = 20


def make_gradient(point_list, point):
    part = math.trunc(point * (len(point_list) - 1))
    if part == (len(point_list) - 1):
        part = len(point_list) - 2
    scale = point * (len(point_list) - 1) - part
    final_point = []
    for i in range(len(point_list[part])):
        left = point_list[part][i]
        right = point_list[part + 1][i]
        diff = right - left
        final_point.append(left + diff * scale)
    return final_point


def plot_color_gradients(gradients, names):
    rc('legend', fontsize=10)
    column_width_pt = 400
    pt_per_inch = 72
    size = (column_width_pt / pt_per_inch) * 1.2

    fig, axes = plt.subplots(nrows=len(gradients), sharex='all', figsize=(size, size))
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.26, right=0.96)

    for ax, gradient, name in zip(axes, gradients, names):
        img = np.zeros((2, samples, 3))
        for i, v in enumerate(np.linspace(0, 1, samples)):
            img[:, i] = gradient(v)
        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)
        ax.tick_params(length=5)
        ax2 = ax.twiny()
        ax2.tick_params(length=5, labeltop=False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)
    fig.savefig('gradients.pdf')


def hsv2rgb(h, s, v):
    c = v * s
    m = v - c
    x = c * (1 - abs(((h / 60) % 2) - 1))

    angle = [
        [c, x, 0],
        [x, c, 0],
        [0, c, x],
        [0, x, c],
        [x, 0, c],
        [c, 0, x],
        [c, x, 0]
    ]

    rgb = angle[math.trunc(h / 60)]
    match = v - c
    rgb = [i + match for i in rgb]
    return rgb


def gradient_rgb_bw(v):
    return make_gradient(rgb_bw, v)


def gradient_rgb_gbr(v):
    return make_gradient(rgb_gbr, v)


def gradient_rgb_gbr_full(v):
    return make_gradient(rgb_gbr_full, v)


def gradient_rgb_wb_custom(v):
    return make_gradient(rgb_wb_custom, v)


def gradient_hsv_bw(v):
    hsv = make_gradient(hsv_bw, v)
    return hsv2rgb(hsv[0], hsv[1], hsv[2])


def gradient_hsv_gbr(v):
    hsv = make_gradient(hsv_gbr, v)
    return hsv2rgb(hsv[0], hsv[1], hsv[2])


def gradient_hsv_unknown(v):
    hsv = make_gradient(hsv_unknown, v)
    return hsv2rgb(hsv[0], hsv[1], hsv[2])


def gradient_hsv_custom(v):
    hsv = make_gradient(hsv_custom, v)
    return hsv2rgb(hsv[0], hsv[1], hsv[2])


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])
