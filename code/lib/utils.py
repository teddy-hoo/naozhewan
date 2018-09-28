# -*- coding: utf-8 -*-


def normalize(cell):
    return (cell - cell.min() + 1) / (cell.max() - cell.min() + 1)
