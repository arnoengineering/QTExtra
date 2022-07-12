import numpy as np
from numpy import linspace
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import *
from functools import partial
import pandas as pd
import sys


# import numpy as np


class gearPlot(pg.GraphicsLayoutWidget):
    def __init__(self, p):
        super().__init__()
        # self.gr = p.ratio

        self.proceses = [('isothermal', 0.5)]
        # legforce could be both

        self.par = p
        self.ts_plot = self.addPlot(1, 0)
        self.pv_plot = self.addPlot(0, 0)
        self._set_ts_plot()
        self._set_pow_plot()
        self.reset_v_cad()
        self._set_pv_p()

    def _set_ts_plot(self):
        self.ts_plot.addLegend()
        self.ts_plot.setLabels(**{'title': 'Ratio per rear gear', 'left': 'ratio', 'bottom': 'rear index'})
        self.ts = self.ts_plot.plot(pen='c', width=3, name='Combined')

    def _set_pv_p(self):
        self.pv_plot.setLabels(
            **{'title': 'cadence vs speed', 'left': ('cad', 'RPM'), 'bottom': ('speed', 'km/h')})
        self.pv = self.pv_plot.plot(pen='c', width=3)
        pass

    def res(self, v, c):
        s1 = v.shape[1]
        z = np.arange(s1)
        z2 = np.arange(c.size)
        for n, i in enumerate(self.g_p):
            i.setData(z, v[n])
        self.ts.setData(z2, c)
        self.res_other()
    #
    # def res_other(self):
    #     # cadence
    #
    #     self.pv.setData(self.par.process['p'], self.par.process['v'])  # asuming df also need to add path
    #
    #     self.ts.setData(s_km, w_t)


class Window(QMainWindow):
    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        self.m_d_s_n = 'Element'
        self.s_d_s_n = 'Type'
        self.settings = QSettings('Chem E', 'Arno')

        # self.wether = {'water': {'p': {'p': [1, 21, 36], 't': [14, 2, 45]},
        #                          't': {'p': [12, 21, 423], 't': [1, 2, 32]}},
        #                'H2': {'p': {'p': [1, 211, 3], 't': [1, 24, 45]},
        #                       't': {'p': [15, 51, 43], 't': [14, 2, 32]}}}  # todo from csv?

        self.setWindowTitle('Thermo Calc')
        self.file = '340 py_noex2.xlsx'
        self.wether = {}
        print('self.load_d')
        self.load_data()
        print('self.set_table')
        self._set_table()
        print('self.set tool')
        self._set_tool()
        print('self.update set')
        self._update_set()
        print('self.reset inputs')
        self.reset_inputs()
        print('self.reset table')
        self.reset_table()
        print('self.init fin')

    def _set_tool(self):
        # pv, st, or sis plots
        # scale

        # todo cell pressed table
        self.tool_bar = QToolBar('Main toolbar')
        self.qual = 0
        self.vari_ls = 'vhs'
        self.varis = SuperText('t2', self, vals=['p', 't'])

        # self.save_op = SuperText('File Options', self, vals=self.button_list)
        print('load main')

        self.main_data_selector = SuperCombo(self.m_d_s_n, self, vals=list(self.wether.keys()), run=False)
        print('load second')
        self.second_data_selector = SuperCombo(self.s_d_s_n, self, run=False)
        print('load combo')

        self.combo = {self.m_d_s_n: self.main_data_selector, self.s_d_s_n: self.second_data_selector}
        self.tool_bar.addWidget(self.main_data_selector.wig)
        self.tool_bar.addWidget(self.second_data_selector.wig)
        self.tool_bar.addWidget(self.varis)

        self.addToolBar(self.tool_bar)

    def load_data(self):
        print('load ex')
        xl = pd.ExcelFile(self.file)

        docs = xl.sheet_names
        for sheet in docs:
            # print(f'Sheet: {sheet}')
            try:
                el, typ = sheet.split('_', 1)
            except ValueError:
                continue
            sh = pd.read_excel(xl, sheet_name=sheet)

            if el not in self.wether:
                self.wether[el] = {}
            self.wether[el][typ] = sh
            # print(f'el: {el}, type: {typ}')
            # print(sh.head())

    def run_cmd(self, i, j=None):
        print(f'run cmd i,j: ({i}, {j})')
        if i == self.m_d_s_n:
            self.reset_inputs(j)
        elif i == self.s_d_s_n:
            self.reset_outputs(j)
        else:
            self.reset_v(i)

    def reset_inputs(self, ii=None):
        print(f'run reset iputs ii: ({ii})')
        if ii is not None:
            print('reset curel')
            self.cur_el = ii
        print(f'second data reset: cur el = ', self.cur_el)
        self.second_data_selector.reset_data()
        print(f'reset outputs')
        self.reset_outputs()

    def reset_outputs(self, jj=None):
        print(f'run reset outputs jj: ({jj})')
        if jj is not None:
            # print('reset curty')
            self.cur_ty = jj
        # print(f'cur data reset: cur ty=', self.cur_ty)
        self.current_data = self.wether[self.cur_el][self.cur_ty]
        # print(f'var reset')
        self.varis.reset_data()
        # todo current data?

        self.reset_table()

    def _set_table(self):
        # qtable?
        self.table = QTableWidget()
        self.setCentralWidget(self.table)

        self.table.cellClicked.connect(self.tab_s)

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
        data_ls = list(self.current_data.columns)
        # if self.cur_ty in 'tp':

        self.table.setHorizontalHeaderLabels(data_ls)
        for n in range(r):
            for m in range(c):
                self.table.setItem(n, m, QTableWidgetItem(str(self.current_data.iloc[n, m])))

    # todo add process so click to add step and move, add add x
    def tab_s(self):
        col_n = self.table.currentColumn()
        r_n = self.table.currentItem().text()

        col = list(self.current_data.columns)[col_n]
        self.varis.but[col].setText(r_n)
        self.reset_v(col,r_n)

    def interp(self, dx, dy, x):
        return self.at_quality(dy, self.get_quality(dx, x))

    # todo two var for superheated,  plot on same t,p the s v curves off the charts
    def point_n(self, n):
        return float(self.varis.but[n].text())

    def set_point(self):
        self.plot_v.plot(self.point_n('v'), self.point_n('p'))
        self.plot_s.plot(self.point_n('s'), self.point_n('t'))
        pass

    def get_quality(self, dx, x):
        return (x - dx[0]) / np.diff(dx)[0]

    def at_quality(self, dy, x):
        return np.diff(dy)[0] * x + dy[0]

    def reset_v(self, i, tex):
        def check_v(iiii):
                v_r = self.at_quality([float(self.varis.but[iiii + f].text()) for f in 'fg'], self.qual)
                t_r = str((round(v_r, 4)))
                self.varis.but[iiii].setText(t_r)
        print('reset vals: i=', i)  # todo x
        # if not t and (x v s h):
        #     highlight and fillin values for vfg, not v
        #     if in midle use those vg vals to solve
        # once both, populat chart
        #         # if cell.clicked:
        #         #     color.hhighlich max 2
        #         # if cell == index:
        #         #     color.highlight
        #         # elif cell == multiindex:
        #         #     for c in cel:
        #         #         color.hightlights

        val = float(tex)
        if i == 'x':  # user input of independant and x gives y or y and x rare test
            self.qual = val
            for ii in self.vari_ls:
                check_v(ii)

        elif i in self.vari_ls:
            he = [float(self.varis.but[i + f].text()) for f in 'fg']  # todo empty
            if min(he)<=val<=max(he):
                self.qual = self.get_quality(he,val)
                for ii in self.vari_ls:
                    if ii != i:
                        check_v(ii)
            else:
                for ii in self.varis.but.keys():
                    if ii != i:

                        self.varis.but[ii].setText('0')
        else:

            # di = self.wether[self.cur_el][self.cur_ty]
            ind_1 = self.current_data[i][1:] > val
            ind_1 = ind_1.values
            index_2 = [ind_1[i] != ind_1[i + 1] for i in range(ind_1.size - 1)]

            ind_3 = index_2.index(True)

            lo = self.current_data.loc[ind_3:ind_3 + 1]
            dx = np.array(lo[i])

            for ii in self.varis.but.keys():

                if ii != i and ii != 'x':
                    if ii in self.vari_ls and self.cur_ty in 'tp':
                        check_v(ii)
                    else:
                        dy = np.array(lo[ii])
                        v_ret = self.interp(dx, dy, val)
                        t = str((round(v_ret, 4)))
                        self.varis.but[ii].setText(t)
                        print(f'ii,i ({ii},{i} set to {t}')
        # self.set_point()
        # todo plot proces on charts

    def _update_set(self):
        print('loading_all')
        self.setting_keys_combo = {self.m_d_s_n: 'water', self.s_d_s_n: 'p'}  # todo current windows, current el

        self.font_ty_default = {}

        self.settings.beginGroup('combo')

        val = self.settings.value(self.m_d_s_n, 'water')
        self.main_data_selector.setCurrentText(val)
        self.cur_el = val
        self.cur_ty = list(self.wether[self.cur_el].keys())[0]
        self.second_data_selector.reset_data()
        val = 'p'  # self.settings.value(self.s_d_s_n,'p')
        self.main_data_selector.setCurrentText(val)
        self.cur_ty = val

        self.settings.endGroup()

        # current_data = self.wether[self.cur_el][self.cur_ty]
        self.current_data = self.wether[self.cur_el][self.cur_ty]
        self.main_data_selector.init_op()
        self.second_data_selector.init_op()
        k = self.settings.allKeys()
        print('res')
        for i, j in [(self.restoreGeometry, "Geometry"), (self.restoreState, "windowState")]:
            if j in k:
                i(self.settings.value(j))

    def user_settings(self, last_ses=True):
        self.settings.setValue("Geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        if last_ses:
            self.settings.beginGroup('combo')

            for ke in self.setting_keys_combo.keys():
                val = self.combo[ke].currentText()
                self.settings.setValue(ke, val)
                print(f'key, val: {ke}, {val}')
            self.settings.endGroup()

    def closeEvent(self, event):

        self.user_settings()
        super().closeEvent(event)


class SuperCombo(QComboBox):
    def __init__(self, name, par, orient_v=True, vals=None, show_lab=True, run=True):
        super().__init__()
        print('user combo')
        self.par = par
        self.orient_v = orient_v
        self.show_lab = show_lab
        self.name = name
        self.wig = QWidget()
        self.lab = QLabel(self.name)
        print('user layout')
        self._layout_set()
        print('vals')
        if vals is not None:
            self.addItems(vals)

        if run:
            print('run')

    def init_op(self):
        print('init: ', self.name)
        self.currentTextChanged.connect(self.rc)
        print('connected')

    def rc(self):
        x = self.currentText()
        print('rc: x: ', x)

        if x is not None and x != '':
            self.par.run_cmd(self.name, x)

    # noinspection PyArgumentList
    def _layout_set(self):
        print(f'super combo: {self.name}, layout set')
        if self.orient_v:
            self.layout = QVBoxLayout()
        else:
            self.layout = QHBoxLayout()
        self.layout.addWidget(self)
        self.layout.addWidget(self.lab)
        self.wig.setLayout(self.layout)

    def reset_show(self, show_lab=False, flip=False):
        if flip:
            self.orient_v = not self.orient_v
            self._layout_set()
        if show_lab:
            self.show_lab = not self.show_lab
            if self.show_lab:
                self.layout.addWidget(self.lab)
            else:
                self.layout.removeWidget(self.lab)

    def reset_data(self):
        try:
            self.currentTextChanged.disconnect(self.rc)
        except TypeError:
            pass
        self.clear()
        self.addItems(list(self.par.wether[self.par.cur_el].keys()))
        if self.par.cur_ty in self.par.wether:
            self.setCurrentText(self.par.cur_ty)
        self.currentTextChanged.connect(self.rc)


class SuperText(QWidget):
    def __init__(self, name, par, orient_v=True, vals=None, show_lab=True):
        super().__init__()

        self.par = par
        self.orient_v = orient_v
        self.show_lab = show_lab
        self.name2 = name
        self.but = {}
        self.lab = QLabel(self.name2)
        self.lab_b = {}
        self.sub_but_ls = {}
        self.sub_but_lab = {}
        self.layout2 = QGridLayout()
        self.setLayout(self.layout2)
        # self.layout2 = QGridLayout()
        # self.layout2 = QGridLayout()
        # self.setLayout(self.layout2)

        # if vals:
        #     for i in vals:
        #         j = QLineEdit()
        #         j.editingFinished.connect(lambda x: self.par.run_cmd(i,x))
        #         self.but[i] = j
        # # for input
        # if 'fg' in name:
        # QLineEdit(name.split('fg'))  #not more
        # QLineEdit(name).setEnabled(false)  # user enabled, can be shown not inputed
        # toolbar 2 gets name

        # self.reset_data()

    def reset_data(self):
        print('reset data')
        vals = list(self.par.current_data.columns)
        if self.par.cur_ty in 'tp':
            vals.extend(['v', 'h', 's', 'x'])  # todo always?

        print('vals: ', vals)
        vm = []
        for i in self.but.keys():
            if i not in vals:
                self.layout2.removeWidget(self.but[i])
                self.layout2.removeWidget(self.lab_b[i])
                vm.append(i)
        for i in vm:
            del self.lab_b[i]
            del self.but[i]

        n = 0
        mi = 0
        for i in vals:
            if i not in self.but.keys():

                j = QLineEdit()
                k = QLabel(i)
                self.but[i] = j
                self.lab_b[i] = k
                if 'f' in i or 'g' in i:
                    m = 2
                    j.setReadOnly(True)
                    ni = mi
                    mi += 1
                else:
                    m = 0
                    ni = n
                    n += 1
                j.editingFinished.connect(partial(self.run_cm, i))
                self.layout2.addWidget(k, m, ni)
                self.layout2.addWidget(j, m+ 1, ni)  # todo add lable for each

    def run_cm(self, i):
        tex = self.but[i].text()
        print(f'________________\ntex={tex}\n-------------------')
        # print(tex)
        self.par.reset_v(i, tex)
        # self._layout_set()

    #
    # def _layout_set(self):
    #     print('layout data')
    #     # self.layout2 = QGridLayout()
    #     n = 0
    #     if self.orient_v:
    #
    #         for i in self.but.keys():
    #             self.layout2.addWidget(self.but[i], 0, n)  # todo add lable for each
    #             n += 1
    #         self.layout2.addWidget(self.lab, 1, 0, 1, n)
    #     else:
    #         for i in self.but.keys():
    #             self.layout2.addWidget(self.but[i], n, 0)
    #             n += 1
    #         self.layout2.addWidget(self.lab, 0, 1, n, 1)
    #
    #     print('add 4')

    # noinspection PyArgumentList
    # def _layout_set(self):
    #     print('layout data')
    #     print('rc before:', self.layout2.rowCount(), self.layout2.columnCount())
    #
    #     print('rc clear:', self.layout2.rowCount(),self.layout2.columnCount() )
    #     n = 0
    #     if self.orient_v:
    #
    #         for i in self.but.keys():
    #             print('add 1')
    #             #
    #
    #             n += 1
    #         # self.layout2.addWidget(self.lab, 2, 0, 1, n)
    #         print('rc add v:', self.layout2.rowCount(), self.layout2.columnCount())
    #         print('add 2')
    #     else:
    #         for i in self.but.keys():
    #             print('add 3')
    #             # self.layout2.addWidget(QLabel(i), n, 0)
    #             self.layout2.addWidget(self.but[i], n, 1)
    #
    #             n += 1
    #         # self.layout2.addWidget(self.lab, 0, 2, n, 1)
    #         print('rc add h:', self.layout2.rowCount(), self.layout2.columnCount())
    #
    #     self.par.update()
    #     print('add 4')

    def reset_show(self, show_lab=False, flip=False):
        if flip:
            self.orient_v = not self.orient_v
            self._layout_set()
        if show_lab:
            print('reset')
            self.show_lab = not self.show_lab
            if not self.show_lab:
                self.layout2.removeWidget(self.lab)


if __name__ == "__main__":
    print('Running Chem E Solve')
    app = QApplication(sys.argv)
    audio_app = Window()
    audio_app.show()
    sys.exit(app.exec_())
