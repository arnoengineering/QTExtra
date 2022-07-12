import numpy as np
# from numpy import linspace
# import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from functools import partial
from PyQt5.QtGui import QColor, QStandardItem
import sys


class keyB(QWidget):
    def __init__(self, par):
        super().__init__()

        self.par = par
        self.qw = ['Qwertyuiop',
                   'Asdfghjkl'
                   'zxcvbnm'
                   ]

        self.input_st = ''
        self.output_st = ''
        self.lay = QGridLayout()
        self.setLayout(self.lay)
        self.col = 6
        self.buts = {}

        self.re_lay = QVBoxLayout()
        self.inp_w = QLabel(self.input_st)  # todo parent
        self.out_w = QLabel(self.input_st)
        self.re_w = QPushButton('Reset')

        self.re_lay.addWidget(QLabel('Input String'))
        self.re_lay.addWidget(self.inp_w)

        self.re_lay.addWidget(QLabel('Output String'))
        self.re_lay.addWidget(self.out_w)

        self.re_lay.addWidget(self.re_w)
        self.re_w.clicked.connect(self.re_v)
        self.re_w.released.connect(self.unset_but)

        for n, row in enumerate(self.qw):
            if n == 2:
                n0 = 4
            else:
                n0 = 4
            for ni, i in enumerate(row):
                j = QPushButton(i)
                self.buts[i] = j
                j.clicked.connect(partial(self.let, i))
                self.lay.addWidget(j, n, n0 + ni * self.col, 1, self.col)

    def let(self, le):
        self.par.light_set(le)

    def unset_but(self):
        self.par.light_unset(self.output_st[-1])


class drums(QWidget):
    def __init__(self):
        super().__init__()
        self.drum_cnt = 3
        self.lay0 = QHBoxLayout()
        self.lay_d = QGridLayout()
        self.lay_in = QGridLayout()
        self.trans_table = [{'A':'g', 'g': 'b'}, {'A':'c', 'd': 'b'}]
        self.tt2 = [[3,1,5]]
        self.curr_tt = []
        self.plug_in = np.array([[0,2],[1,25]])
        self.plag_user = np.array([[0,2],[1,25]])
        """evevelet orentarions at index 0, ie a=0,b=1, thus for {a,c;b,f}-->{z,b;a,e}
        at nth pos clicking y times
        tt2[n-y]=tt2[n]-y  # dont do inplace,
        or np.roll(tt2-y, y)"""

    def _set_drum(self):
        self.drum_pos = []
        self.drum_n = []
        self.cur_pos = []

        self.cur_d = []
        self.ls = ['Drum: ' + str(ni+1) for ni in range(5)]

        self.lay_d.addWidget(QLabel('Current settings'), 0, 0, 1, 3)
        for n in range(self.drum_cnt):
            j = QSpinBox()
            j.setRange(0,25)
            j.setSingleStep(1)
            j.editingFinished.connect(partial(self.user_in, n))

            j.clicked.connect(partial(self.step_drum, n))
            self.cur_pos.append(j)
            self.lay_d.addWidget(j, 1, n)

            k = QComboBox()

            k.currentTextChanged.connect(partial(self.swap_d, n))

            #idx = k.model()
            k.setItemData()
            # for i in self.ls:
            #     it = QStandardItem(i)
            #     it.setBackground(QColor(i))
            #     # self.addItem(it)  # todo value from index
            #
            #     idx.appendRow(it)
            # self.setStyleSheet("background-color: " + self.currentText() + "; }")
            #
            # self.currentTextChanged.connect(self.current_changed)
            # self.cur_d.append(k)

    def current_changed(self):
        v = self.currentText()
        va = self.ls.index(v)

    def user_in(self, n):
        v = self.cur_pos[n].text()
        self.cur_pos[n].setValue(int(v))

    def step_drum(self, d):
        # todo if user init, reset else just index
        pass

    def swap_d(self, d):
        nd = self.cur_d[d].currentText()
        for i in range(3):
            if i != d:
                self.cur_d[i].setItemData(i, False, -1)  # todo find based on other, stylesheet, no click

    def index_drum(self,d):
        # todo combine with stepdrum?
        self.drum_pos[d] = (self.drum_pos[d] + 1) % 26  # todo func
        self.curr_tt[d] = np.roll(self.tt2,self.drum_pos[d]) - self.drum_pos[d]  # todo anim
        if self.drum_pos[d] == 0 and d != 2:
            self.drum_pos[d + 1] = (self.drum_pos[d] + 1) % 26  # flip all first then do or split index and flip funcs, todo reverse

    def app_num_drum(self,n):  # todo plugboard then this, then plugboard
        for i in self.curr_tt:
            n = i[n]
        # n2 = where n in np araay of list if 2 give 1,m ...
        n = self.plug_swap(n)
        for i in np.fliplr(self.curr_tt):
            n = i.index(n)
        return n

    def plug_swap(self, n):  # todo same for external plug
        value_ar = np.argwhere(self.plug_in == n)
        # should be tuple with opsite side
        n_v = self.plug_in[value_ar[0], (self.plug_in[1] + 1) % 2]
        return n_v

    def set_val(self, vals):
        for n in range(3):
            self.cur_pos[n].setValue(vals[n])

# class plugbard
class plugboard(QWidget):
    def __init__(self):
        super().__init__()
        self.but = []
        self.lay = QGridLayout()
        self.pairs = np.empty((10,2))
        self.active_pair = None

        self.te = 'abcdefghijklmnopqrstuvwxyz'
        r = 0
        n = 0
        for i in self.te:
            j = QPushButton(i)
            self.but.append(j)
            j.clicked.connect(partial(self.plug_sp, i))
            self.lay.addWidget(j, r, n)
            n += 1
            if n > 12:
                n = 0
                r += 1

    def plug_sp(self, pl):
        # todo on error add popup
        if pl in self.pairs: # clicked
            if self.acttive_pair:
                if pl in self.pairs[self.active_pair]:  # todo on none
                    # meaning it was clicked last step
                    self.pairs[self.active_pair, 0] = 0  # todo a, n
                    self.active_pair = None
                    # todo reset color, pick from list
                else:
                    # error
                    pass
            else:
                ind = np.argwhere(self.pairs == pl)
                self.active_pair = ind[0]
                self.pairs[ind[0], 0] = self.pairs[ind[0],ind[1]]  # todo better way?
                self.pairs[ind[0], 1] = 0
                # todo reset color, pick from list
        else:
            if self.active_pair:
                self.pairs[self.active_pair, 1] = pl
                self.active_pair = None
                # todo color, pick from list

            else:
                open_v = 10  # todo replace with empty index
                self.active_pair = open_v
                self.pairs[open_v, 0] = pl
                # todo set color


class Window(QMainWindow):
    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()

        self.changed_drums = False

        self.cen = keyB()
        self.setCentralWidget(self.cen)

        self.plug_d = QDockWidget('Plug Board')
        self.addDockWidget(Qt.BottomDockWidgetArea, self.plug_d)
        self.plug_w = plugboard()
        self.plug_d.setWidget(self.plug_w)

        self.dr_d = QDockWidget('Drums')
        self.addDockWidget(Qt.TopDockWidgetArea, self.dr_d)
        self.d_w = drums()
        self.dr_d.setWidget(self.d_w)

    def keyPressEvent(self, event):
        te = event.text().lower()
        if te in ''.join(self.cen.qw):
            self.l_trig(te)

    def keyReleaseEvent(self, event):
        pass  # todo on button relese do same

    def re_v(self):
        if self.changed_drums:
            drums.setValues(self.int_vals)
        else:
            drums.set_vale(0, 0, 0)
        self.cen.input_st = ''
        self.cen.output_st = ''

    def light_set(self, k):
        self.cen.buts[k].setStyleSheet('background-color: yellow};')

    def light_unset(self, k):
        self.cen.buts[k].setStyleSheet('background-color: lightgrey};')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    audio_app = Window()
    audio_app.show()
    sys.exit(app.exec_())