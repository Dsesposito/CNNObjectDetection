import json
import multiprocessing
import random
import string
from functools import partial

import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from utils import DataLabel


class DataSetHandler(object):
    def __init__(self, store_images=False):
        self.train = DataLabel()
        self.test = DataLabel()
        self.val = DataLabel()
        self.amt_classes = None
        self.classes_name = None
        self.width = 100
        self.dpi = 50
        self.store_images = store_images
        self.input_path = join(os.getcwd(), 'data', 'ndjson')
        self.output_path = join(os.getcwd(), 'data', 'quick_draw_images')

    def build_data_set(self):
        files = [file for file in listdir(self.input_path) if isfile(join(self.input_path, file))
                 and file != '.gitignore']
        for index, file in enumerate(files):
            self.build_one_class_dataset(file)

    def build_one_class_dataset(self, file):
        amt_train_images = 10000
        amt_test_images = 1000
        amt_val_images = 1000
        amt_tot = amt_train_images + amt_test_images + amt_val_images

        file_name = os.path.splitext(file)[0]
        class_name = file_name.split('_')[-1]

        path = join(self.output_path, class_name)
        if os.path.exists(path):
            return

        os.mkdir(path)

        drawings_strokes = []
        with open(os.path.join(self.input_path, file)) as f:
            for line_index, line in enumerate(f):
                threshold = 2 * amt_tot
                if line_index >= threshold:
                    break
                json_line = json.loads(line)
                drawings_strokes.append(json_line['drawing'])

        pool = multiprocessing.Pool(processes=4)
        func = partial(self.strokes_to_ndarray, class_name)
        pool.map(func, drawings_strokes)
        pool.close()
        pool.join()

    def strokes_to_ndarray(self, class_name, strokes):
        fig = plt.figure(
            figsize=(self.width / self.dpi, self.width / self.dpi), dpi=self.dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        for stroke in strokes:
            x = stroke[0]
            y = stroke[1]
            plt.plot(x, y, c='k')

        plt.gca().invert_yaxis()
        path = join(self.output_path, class_name)
        file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        plt.savefig(os.path.join(path, file_name), bbox_inches='tight', pad_inches=0, dpi=self.dpi)
        plt.close()
        return
