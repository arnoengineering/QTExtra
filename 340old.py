
import numpy as np
from numpy import linspace
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from functools import partial
import pandas as pd
import sys


# import numpy as np


class gearPlot(pg.GraphicsLayoutWidget):
    def __init__(self, p):
        super().__init__()
        # self.gr = p.ratio

        self.input_v = {'WD': 0.7, 'Pedal Rad': 0.13, 'Wind': 0}
        # legforce could be both
        self.depend_v = {'Ratio': p.ratio}

        self.ambi_v = {'Leg Force': 1000, 'Cadence': 80, 'Ground Speed': 50, }

        self.par = p
        self.gear_plot = self.addPlot(1, 0)
        self.cadence_plot = self.addPlot(0, 0)
        self.torque_plot = self.addPlot(0, 1)
        self.air_plot = self.addPlot(1, 1)
        self._set_g_plot()
        self._set_pow_plot()
        self.reset_v_cad()
        self._set_cad_p()
        self._set_t_p()

    def _set_g_plot(self):
        self.gear_plot.addLegend()
        self.gear_plot.setLabels(**{'title': 'Ratio per rear gear', 'left': 'ratio', 'bottom': 'rear index'})
        self.g_p = []
        col = ['r', 'g', 'b', 'y']
        for i in range(len(self.par.gear[0])):
            st = 'Front chainring: ' + str(i)
            self.g_p.append(self.gear_plot.plot(pen=col[i], width=3, name=st))
        self.combine_p = self.gear_plot.plot(pen='c', width=3, name='Combined')

    def _set_t_p(self):
        self.torque_plot.setLabels(
            **{'title': 'Torque vs speed', 'left': ('Torque', 'Nm'), 'bottom': ('speed', 'km/h')})
        self.tor_s = self.torque_plot.plot(pen='c', width=3, name='Speed')
        pass

    def _set_cad_p(self):

        self.cadence_plot.setLabels(
            **{'title': 'cadence vs speed', 'left': ('cad', 'RPM'), 'bottom': ('speed', 'km/h')})
        self.cad = self.cadence_plot.plot(pen='c', width=3)
        pass

    def _set_pow_plot(self):
        self.air_plot.addLegend()
        self.air_plot.setLabels(**{'title': 'Air resisrace vs speed', 'left': ('power', 'W'),
                                   'bottom': ('speed', 'km/h')})
        self.na = self.air_plot.plot(pen='c', width=3, name='No resestance')
        self.air_r = self.air_plot.plot(pen='b', width=3, name='Air')

    def res(self, v, c):
        s1 = v.shape[1]
        z = np.arange(s1)
        z2 = np.arange(c.size)
        for n, i in enumerate(self.g_p):
            i.setData(z, v[n])
        self.combine_p.setData(z2, c)
        self.res_other()

    def res_other(self):
        # cadence
        s = linspace(0, 20)

        # air

        power = 150 + 17.5 * s
        w_p = []
        w_t = []
        w_cad = []
        for si in range(s.size):
            self.ambi_v['Ground Speed'] = s[si]
            self.reset_air(power[si])
            w_p.append(self.depend_v['Pow'])
            self.reset_air(power[si], False)
            w_t.append(self.depend_v['Torque'])
            w_cad.append(self.ambi_v['Cadence'])

        s_km = s * 3.6
        self.air_r.setData(s_km, w_p)
        self.na.setData(s_km, power)

        self.cad.setData(s_km, w_cad)

        self.tor_s.setData(s_km, w_t)

    def reset_pre(self):
        self.depend_v['WR'] = self.input_v['WD'] / 2
        self.depend_v['WC'] = self.input_v['WD'] * np.pi

    def reset_v_cad(self):
        self.reset_pre()
        # pre defined

        self.depend_v['Omega'] = self.ambi_v['Cadence'] * np.pi / 30
        self.depend_v['Torque'] = self.ambi_v['Leg Force'] * self.input_v['Pedal Rad']
        self.depend_v['W Omega'] = self.depend_v['Omega'] * self.depend_v['Ratio']
        self.depend_v['W M'] = self.depend_v['Torque'] / self.depend_v['Ratio']
        self.ambi_v['Ground Speed'] = self.depend_v['W Omega'] * self.depend_v['WC']
        self.depend_v['Ground Force'] = self.depend_v['W M'] / self.depend_v['WR']

        # ||wheel tor=fric*wrad||power, vs no wind"""

    def reset_v_ground(self):
        self.reset_pre()
        self.depend_v['W Omega'] = self.ambi_v['Ground Speed'] / self.depend_v['WC']
        self.depend_v['Omega'] = self.depend_v['W Omega'] / self.depend_v['Ratio']
        self.ambi_v['Cadence'] = self.depend_v['Omega'] * 30 / np.pi
        self.depend_v['Pow'] = self.depend_v['Torque'] * self.depend_v['Omega']

    def reset_air(self, resistace, drag=True):
        v = self.ambi_v['Ground Speed'] - self.input_v['Wind']
        a = 2  # m^2
        c = 1
        if drag:
            wind_drag = 0.5 * 1.255 * a * c * v ** 2
        else:
            wind_drag = 0
        self.reset_v_ground()
        self.depend_v['Ground Force'] = wind_drag
        # print(f'power n: {resistace}, drag: {wind_drag}, total: {self.depend_v["Ground Force"]}')
        self.depend_v['W Omega'] = self.ambi_v['Ground Speed'] / self.depend_v['WC']
        self.depend_v['Omega'] = self.depend_v['W Omega'] / self.depend_v['Ratio']
        self.depend_v['W M'] = self.depend_v['Ground Force'] * self.depend_v['WR']
        self.depend_v['Torque'] = self.depend_v['Ratio'] * self.depend_v['W M']
        self.depend_v['Pow'] = self.depend_v['Torque'] * self.depend_v['Omega'] + resistace

    def current_n(self):

        self.reset_v_cad()  # since no values changed just cal any


class Window(QMainWindow):
    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        self.gear = [[34, 52],
                     [11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 28]]
        self.active_g = np.ones(2)
        self.ratio = 1
        self.wether = {'water': {'p':{'p': [1, 21, 36], 't':[14,2,45]},
                                 't':{'p': [12, 21, 423], 't':[1,2,32]}},
                       'H2':{'p':{'p': [1, 211, 3], 't':[1,24, 45]},
                             't':{'p': [15, 51, 43], 't':[14,2,32]}}} # todo from csv?
        other_types = {}

        self.setWindowTitle('QMainWindow')
        self.cen = QWidget()
        self.scale_n = [[10, 2], [10, 3]]

        self.setCentralWidget(self.cen)

        # # self.tb = QToolBar(self)
        # self.addToolBar(self.tb)
        #self.da = QAction('Set Gear')
        # self.tb.addAction(self.da)
        # self.da.triggered.connect(self.dia)
        self.varis = {}
        self.cur_el = 'water'
        self.cur_ty = 'p'
        current_data = self.wether[self.cur_el][self.cur_ty]
        self.current_data = pd.DataFrame.from_dict(current_data)
        self._set_table()
        self._set_tool()

        # self._set_out()
        # self._set_data()
        # self._set_tar()

    # def _set_sub_win(self):
    #
    #     self.third = QWidget()
    #     self.sub_dock = QDockWidget('Tirtiary')
    #     self.sub_dock.setWidget(self.third)
    #     self.addDockWidget(Qt.RightDockWidgetArea, self.sub_dock)
    #
    #     self.sub_d = {}
    #     self.su_lay = QGridLayout()
    #     for n, i in enumerate(list(self.current_data.columns)):
    #         j = QLineEdit()
    #         k = QLabel(i)
    #         self.sub_d[i] = j
    #         self.su_lay.addWidget(k,n,0)
    #         self.su_lay.addWidget(j,n,1)
    #         j.editingFinished.connect(self.reset_v)
        pass
    def _set_tool(self):
        # pv, st, or sis plots
        # scale

        # todo cell pressed table

        self.select_lay = QGridLayout()

        self.main_data_selector = QComboBox()
        self.second_data_selector = QComboBox()

        self.second_data_selector.currentTextChanged.connect(self.reset_outputs)

        self.select_lay.addWidget(QLabel('Element'), 0, 0)
        self.select_lay.addWidget(QLabel('Type'), 0, 1)
        self.select_lay.addWidget(self.main_data_selector, 1, 0)
        self.select_lay.addWidget(self.second_data_selector, 1, 1)


        for i in self.wether.keys():  # todo set default secondart
            self.main_data_selector.addItem(i)

        self.main_data_selector.currentTextChanged.connect(self.reset_inputs)
        self.reset_inputs()

    def reset_inputs(self):  # todo hold state if sme dont change., also reset other?, set charts
        print('Hi')
        self.cur_el = self.main_data_selector.currentText()
        self.second_data_selector.currentTextChanged.disconnect(self.reset_outputs)
        self.second_data_selector.clear()
        for i in self.wether[self.cur_el].keys():
            self.second_data_selector.addItem(i)
        if self.cur_ty in self.wether:

            self.second_data_selector.setCurrentText(self.cur_ty)
        self.second_data_selector.currentTextChanged.connect(self.reset_outputs)
        self.reset_outputs()

    def reset_outputs(self):
        print('Hi2')
        self.varis = {}  # todo units, on wrong input popup: riqure two? and p for comp, set table

        self.var_layout = QGridLayout()
        self.cur_ty = self.second_data_selector.currentText()
        self.current_data = pd.DataFrame.from_dict(self.wether[self.cur_el][self.cur_ty])
        # todo current data?
        for n, i in enumerate(list(self.current_data.columns)):
            print('j', i)
            self.var_layout.addWidget(QLabel(i),n, 0)
            j = QLineEdit()
            j.editingFinished.connect(partial(self.reset_v, i))
            self.var_layout.addWidget(j, n, 1)
            self.varis[i] = j

        self.layout = QVBoxLayout()

        self.layout.addLayout(self.select_lay)
        self.layout.addLayout(self.var_layout)
        self.cen.setLayout(self.layout)
        self.reset_table()

    def _set_table(self):
        # qtable?
        self.table = QTableWidget()
        self.table_dock = QDockWidget('Table')
        self.table_dock.setWidget(self.table)
        self.addDockWidget(Qt.RightDockWidgetArea, self.table_dock)

        self.table.cellClicked.connect(self.tab_s)
        self.reset_table()

    def _set_plots(self):
        # plot_s = self.current_data['S', 'T'].sort('S')
        # plot_v = self.current_data['v', 'p'].sort('v')
        # plot(plot_s['s'],plot_s['T'])
        # plot(plot_v['v'], plot_s['p'])
        pass

    def reset_table(self):
        self.table.clear()
        r, c = self.current_data.shape
        # r = max(len(x) for x in self.current_data.values())
        self.table.setRowCount(r)
        self.table.setColumnCount(c)
        self.table.setHorizontalHeaderLabels(list(self.current_data.columns))
        for n in range(r):
            for m in range(c):
                self.table.setItem(n, m, QTableWidgetItem(str(self.current_data.iloc[n,m])))

    # todo add process so click to add step and move, add add x
    def tab_s(self):
        col_n = self.table.currentColumn()
        r_n = self.table.currentItem().text()

        col = list(self.current_data.columns)[col_n]
        self.varis[col].setText(r_n)
        self.reset_v(col)

    def interp(self, dx, dy, x):
        return self.at_quality(dy, self.get_quality(dx, x))

    # def interp_dual(self):
    #
    #     pass

    # todo two var for superheated,  plot on same t,p the s v curves off the charts
    def point_n(self, n):
        return float(self.varis[n].text())

    def set_point(self):
        # plot_v.plot(self.point_n('v'),self.point_n('p'))
        # plot_s.plot(self.point_n('s'), self.point_n('t'))
        pass

    def get_quality(self, dx, x):
        return (x - dx[0]) / np.diff(dx)[0]

    def at_quality(self, dy, x):
        return np.diff(dy)[0] * x + dy[0]

    def reset_v(self, i):  # todo quality, if inbetween s, [if decresing?
        val = float(self.varis[i].text())

        # di = self.wether[self.cur_el][self.cur_ty]
        ind_1 = self.current_data[i] > val  # todo will Pm care, see below?
        # todo do true false type and check for change, ::: above dont think so
        ind_1 = ind_1.values
        index_2 = [ind_1[i] != ind_1[i+1] for i in range(ind_1.size-1)]
        # sort [1 2 3 4 5] > 3.5
        #      [F F F T T]
        #        [F F T F]
        # sort [5 4 3 2 1] > 3.5
        #      [T T F F F]
        #        [F T F F]
        ind_3 = index_2.index(True)

        lo = self.current_data.loc[ind_3:ind_3+1]
        dx = np.array(lo[i])
        for ii in self.varis.keys():
            if ii != i:
                dy = np.array(lo[ii])
                v_ret = self.interp(dx, dy, val)
                self.varis[ii].setText(str((round(v_ret,4))))
        self.set_point()
        # todo plot proces on charts



if __name__ == "__main__":
    app = QApplication(sys.argv)
    audio_app = Window()
    audio_app.show()
    sys.exit(app.exec_())
