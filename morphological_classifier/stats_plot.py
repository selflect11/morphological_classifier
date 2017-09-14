# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from morphological_classifier import constants
from itertools import product as cartesian_product


class StatsPlotter:
    def __init__(self, plot_name, plotting):
        self.plot_name = plot_name if plot_name else 'confusion_matrix'
        self.plotting = plotting

    def plot_confusion_matrix(self, cm,
                              normalize=False,
                              title='Confusion Matrix',
                              cmap=plt.cm.Blues):
        if not self.plotting:
            return

        classes = sorted(constants.TAGS)
        n_classes = len(classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2
        fmt = '.2f'
        for i, j in cartesian_product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt).rstrip('0').rstrip('.'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=5)

        plt.tight_layout()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.savefig(str.join('', ['conf_matrices/',
                    self.plot_name]) + '.png',
                    dpi=200)
        plt.draw()
        #plt.show()
