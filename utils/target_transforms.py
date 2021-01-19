import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):
    """
    For CNNLSTM
    """

    def __call__(self, target):
        return [target['last'], target['current'], target['next']]


class FlowLabel(object):
    """
    For Guided Train
    """

    def __call__(self, target):
        target = target.split('_')[1:]
        return list(map(int, target))


class VideoID(object):

    def __call__(self, target):
        return target['video_id']
