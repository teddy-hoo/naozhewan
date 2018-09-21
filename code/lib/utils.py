# -*- coding: utf-8 -*-


def normalize(cell):
    return (cell - cell.min()) / (cell.max() - cell.min())

