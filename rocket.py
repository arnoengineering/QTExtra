import sys

import numpy as np
import PyQt5.QtWidgets as QtW
from PyQt5.QtWidgets import *
# QToolBar, QStatusBar
import os
from PyQt5.QtCore import Qt, QTimer, QPointF, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QFont

from numpy import sin, cos, pi, fft
import time
from functools import partial


class Point:
    def __init__(self, p):
        self.size = 2
        self.loc = p
        self.color = 'green'


class line:
    def __init__(self, p):
        self.size = 1
        self.loc = p
        self.color = 'green'

    def tangent(self):
        pass


class circ:
    def __init__(self, p1, p2,p3):
        self.points = [p1, p2, p3] # todo point, pdraw on cration, todo select multiple, todo np sincos
        self.rad = None
        self.cen = None

    def find_rad(self):
        p3_1 = self.points[2]-self.points[0]
        p2_1 = self.points[1]-self.points[0]
        th = np.arctan2(*p2_1)
        th_3 = np.arctan2(*p3_1)
        phi = pi/2-th_3+th
        l3 = np.linalg.norm(p3_1)
        self.rad = l3/(2*cos(phi))
        th_4 = phi-th_3
        self.cen = np.array((cos(th_4),sin(th_4)))*self.rad


class gear:
    def __init__(self):
        self.d = 20
        self.r = self.d/2
        self.db = self.d*cos(pi/9)
        self.rb = self.db/2
        self.r_fill = 2
        self.fill_cent = None
        self.fill_p = None
    # def invol(self,th):
    #     return

    def rot(self, th):
        return np.array((cos(th), sin(th)))

    def find_fill(self):
        qf = [self.rb**2, self.rb*self.r_fill, self.r**2+self.r*self.r_fill +self.rb**2]
        j = np.roots(qf)
        th = np.real(j[0])
        th_n = np.arccos((self.r+self.r_fill)/self.rb)  # todo use fuction of invol, todo dedend
        n = th_n - th
        self.fill_cent = self.rot(-n)

        fill_th = np.linspace(pi/2+th,pi-n)
        self.fill_p = self.r_fill*self.rot(fill_th)
        # from cent to fil: db,dr,todo use ether, for each discrip add point
        # then fill
        # then invol till outsid
        # then ouside
        # dash rest

class CannyEdge(QWidget):
    def __init__(self):
        super().__init__()
        print('\n___________\nCanny Edge Widget\n___________')

        self.setMinimumSize(200, 200)
        self.tool = 'line'
        self.lp = []
        self.sizes = (200, 200)
        self.initial_image = np.zeros(self.sizes)
        self.undrawn = QImage(self.size(), QImage.Format_RGB32)
        self.lastPoint = QPoint()
        self.setMouseTracking(True)
        self._set_image_clear()
        self._set_brush()

    def _set_brush(self):
        print('setting Brush')

        self.active = False
        self.brush_size = 10
        self.r = 2  # todo scale to iomg size,
        print('sett Brush')

    def _set_image_clear(self):
        print('setting img clear')
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        # self.image = QPixmap(img)
        # self.setPixmap(self.image)
        print('sett img clear')

    def set_image(self, arr):
        print('setting img')
        self.initial_image = arr
        self.image = QImage(arr)
        # self.image = QPixmap(img)
        # self.setPixmap(self.image)
        print('sett img')

    def set_img_from_file(self, file):
        pass

    def scale_image(self):
        pass
        pass

    def img_to_array(self):
        '''  Converts a QImage into an opencv MAT format  '''

        img = self.image.convertToFormat(QImage.Format.Format_RGB32)
        width = img.width()
        height = img.height()

        ptr = img.constBits()
        arr = np.array(ptr).reshape((height, width, 4))  # Copies the data

    # method for checking mouse cicks
    # paint event
    def mousePressEvent(self, event):
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            self.active = True
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):  # hold and drop todo point and drop, save line tthen reset if point moves
        # todo for arc, 3point arc, p1-p2, rad = (p1+p2)/2
        # print('size: ', self.size())
        self.undrawn.fill(Qt.black)
        vect_painter = QPainter(self.undrawn)
        painter = QPainter(self.image)
        rad = self.brush_size // 2
        if any(self.drawing):
            if event.buttons() & Qt.LeftButton:
                if self.drawing[0]:
                    col = Qt.white

                else:
                    col = Qt.black

                # painter.setBrush(col)
                print('set pen')
                painter.setPen(QPen(col, self.brush_size))  # Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin

                # draw line from the last point of cursor to the current point
                # this will draw only one step
                print('drawp')
                painter.drawLine(self.lastPoint, event.pos())

                # change the last point
                print('set last_p')
                # painter.drawEllipse(event.pos(), rad, rad)
                vect_painter.drawImage(self.undrawn.rect(), self.image, self.image.rect())

            # method for mouse left button release
            # update
        self.update()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.active = False


    def paintEvent(self, event):
        # create a canvas
        # if self.active:
        canvasPainter = QPainter(self)

        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect(), self.undrawn, self.undrawn.rect())

    def resizeEvent(self, event):
        self.undrawn = self.undrawn.scaled(self.size())
        self.image = self.image.scaled(self.size())
