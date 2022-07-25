import numpy as np
from numpy import linspace
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import *
from functools import partial
import pandas as pd
import sys


# import numpy as np

class powerCycle:
    def __init__(self):
        pass


class vaporCycle(powerCycle):
    def __init__(self):
        super().__init__()
        pass


class Rankine(powerCycle):
    def __init__(self):
        super().__init__()
        self.process = ['isentropic compress', 'isobaric heat addition', 'isentropic expand', 'isobaric heat reject']
        # todo const pressure
        # todo mix
        # todo reheat
        pass


class gasCycle(powerCycle):
    def __init__(self):
        super().__init__()
        pass


class brayton(gasCycle):
    def __init__(self):
        super().__init__()
        self.proces_ls = ['isentropic compress', 'isobaric heat reject', 'isentropic expand', 'isobaric heat addition']

    def data(self, **kwargs):
        p1 = 0
        p4 = p1
        p2 = 0
        ex = 't2/t1=(p2/p1)^((k-1)/k)'
        p1 = p2


class carnotCycle(powerCycle):
    def __init__(self):
        super().__init__()


class gearPlot(pg.GraphicsLayoutWidget):
    def __init__(self, p):
        super().__init__()
        # self.gr = p.ratio

        self.proceses = [('isothermal', 0.5)]
        # legforce could be both

        self.par = p
        self.data = pd.DataFrame()
        self.ts_plot = self.addPlot(1, 0)
        self.pv_plot = self.addPlot(0, 0)

        self._set_ts_plot()
        self._set_pv_p()
        self.p = [self.ts, self.pv, self.tv]
        self.ps = [self.ts_points, self.pv_points, self.tv_points]

    def _set_ts_plot(self):
        self.ts_plot.addLegend()
        self.ts_plot.setLabels(**{'title': 'Teperature vs entropy', 'left': ('T', 'K'), 'bottom': ('S', "kJ/(kgK)")})
        self.ts = {'f': self.ts_plot.plot(pen='c', width=3), 'g': self.ts_plot.plot(pen='c', width=3)}
        self.ts_points = pg.ScatterPlotItem(pen='r', width=3)
        self.ts_plot.addItem(self.ts_points)

    def _set_pv_p(self):
        self.pv_plot.setLabels(
            **{'title': 'Preasure vs volume', 'left': ('P', 'kPa'), 'bottom': ('v', 'm3/(kg)')})
        self.pv = {'f': self.pv_plot.plot(pen='c', width=3), 'g': self.pv_plot.plot(pen='c', width=3)}
        self.tv = {'f': self.pv_plot.plot(pen='g', width=3), 'g': self.pv_plot.plot(pen='g', width=3)}
        self.pv_points = pg.ScatterPlotItem(pen='r', width=3)
        self.tv_points = pg.ScatterPlotItem(pen='y', width=3)
        self.pv_plot.addItem(self.pv_points)
        self.pv_plot.addItem(self.tv_points)
        # self.pv_plot.setAxisLimits()

    def res(self, points):
        # find plots at sat p, t for lots
        jk = ['Ts', 'Pv', 'Tv']
        for i in range(len(jk)):
            da = []
            da_y = []
            for ik in points:
                da.append(ik[0][jk[i][1]])
                da_y.append(ik[0][jk[i][0]])

            self.ps[i].setData(da, da_y)  # todo next point on f or g or other

        self.par.proces_lines_sat(points)

    # plot point p,v,st, and add line for next point folow line then along curve
    # def res_other(self):
    #     # cadence
    #
    #     self.pv.setData(self.par.process['p'], self.par.process['v'])  # asuming df also need to add path
    #
    #     self.ts.setData(s_km, w_t)

    def reset_el(self):
        # find plots at sat p, t for lots
        jk = ['Ts', 'Pv', 'Tv']
        data = self.par.wether[self.par.cur_el]['t']  # todo on each point add line on each and a few, TOP TEN PERCENT
        # TODO  for gas
        data = data[data.index > 0]
        if self.par.cur_el in ['R134a', 'water']:  # todo for gas, todo find pr, vr z k
            # todo save data
            for i in range(len(jk)):
                for j in 'fg':
                    self.p[i][j].setData(np.array(data[jk[i][1] + j]), np.array(data[jk[i][0]]))

    def proces_lines_ideal_gas(self, points):
        # find plots at sat p, t for lots

        pass

    # def proces_lines_sat(self, points):
    #     jk = ['Ts', 'Pv', 'Tv']
    #
    #     for i in range(len(jk)):
    #         da = []
    #         if 'p' not in jk[i]:
    #             con = 'p'
    #         else:
    #             con = 'T'
    #
    #         da_y = []
    #         for ij in range(len(points)):  # todo invalid, process
    #             # todo type
    #             ik = points[ij][0]
    #             ty = points[ij][1]
    #             p2 = points[ij][0]
    #             if ty == 'compressed' or ik == 'superheated':
    #                 # if compre
    #                 curve = self.par.wether[self.par.current_el][ty].where[con == ik[con]]  # list points
    #             inp = self.par.current_data.where[con == ik[con]]  # start end to poinmt in secter
    #             da.append(ik[jk[i][1]])
    #             da_y.append(ik[jk[i][0]])
    #
    #         self.ps[i].setData(da, da_y)  # todo next point on f or g or other
    #     pass


class Window(QMainWindow):
    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        self.m_d_s_n = 'Element'
        self.s_d_s_n = 'Type'
        self.jk = ['Ts', 'Pv', 'Tv']
        self.settings = QSettings('Chem E', 'Arno')

        # self.wether = {'water': {'p': {'p': [1, 21, 36], 't': [14, 2, 45]},
        #                          't': {'p': [12, 21, 423], 't': [1, 2, 32]}},
        #                'H2': {'p': {'p': [1, 211, 3], 't': [1, 24, 45]},
        #                       't': {'p': [15, 51, 43], 't': [14, 2, 32]}}}  # todo from csv?

        self.setWindowTitle('Thermo Calc')
        self.file = '340 py_noex2.xlsx'
        self.wether = {}
        self.active_varable = []
        self.plot = gearPlot(self)
        self.po = point_ls(self)
        self.proces = ['isothermic', 'isobaric', 'isothermic', 'isobaric']
        self.proscess_mean = {'isentropic': 's', 'isobaric': 'P', 'isochoric': 'v', 'isothermic': 'T'}

        self.load_data()
        self._set_table()
        self._set_tool()
        self._update_set()
        self.reset_inputs()
        self.reset_table()
        print('self.init fin\n----------------------------\n-------------------------------\n---------------\n\n')

    def _set_tool(self):
        print('-set tool')
        # pv, st, or sis plots
        # scale

        # todo cell pressed table
        self.tool_bar = QToolBar('Main toolbar')
        self.qual = 0
        self.vari_ls = 'vhs'
        self.varis = SuperText('t2', self)
        self.d2 = QDockWidget('plot')
        self.d2.setWidget(self.plot)
        self.addDockWidget(Qt.RightDockWidgetArea, self.d2)

        # self.save_op = SuperText('File Options', self, vals=self.button_list)
        self.main_data_selector = SuperCombo(self.m_d_s_n, self, vals=list(self.wether.keys()), run=False)
        self.second_data_selector = SuperCombo(self.s_d_s_n, self, run=False)

        self.combo = {self.m_d_s_n: self.main_data_selector, self.s_d_s_n: self.second_data_selector}
        self.add_but = QPushButton('Update Point')
        self.add_but.clicked.connect(self.reset_point)
        self.tool_bar.addWidget(self.main_data_selector.wig)
        self.tool_bar.addWidget(self.second_data_selector.wig)
        self.tool_bar.addWidget(self.varis)
        self.tool_bar.addWidget(self.add_but)

        self.addToolBar(self.tool_bar)

    def reset_point(self):
        self.active_varable = []
        pass

    def interp_data(self, data, val, var, ty='t'):
        def x23(x):
            y = np.abs((x - val) / val)
            print(f'x23: val: {val}, x: {x}, res: {y}')
            return y < 0.01

        print('-interp data')
        print(f'val: {val}, var: {var}, ty: {ty}')
        print('data =')
        print(data.head())
        if ty == 't':  # todo super x

            # ind_1 = data[var][1:] > val
            # ind_1 = ind_1.values
            # index_2 = [ind_1[i] != ind_1[i + 1] for i in range(ind_1.size - 1)]
            #
            # ind_3 = index_2.index(True)
            #
            # lo = data[1:].loc[ind_3 + 1:ind_3 + 2]
            # # if self.cur_ty not in 'tp':
            # #     if i == list(self.varis.but.keys())[0]:
            #
            # dx = np.array(lo[var])
            # dd = {}
            # print('solving interp ', dx)
            # print('lo = ')
            # print(lo)
            # for ii in data.columns:
            #     if ii != var:
            #         dy = np.array(lo[ii])
            #         v_ret = self.interp(dx, dy, val)
            #         dd[ii] = [round(v_ret, 4)]
            return self.reset_vs(var,val,data[1:])
        else:
            data2 = data[1:]
            data2 = data2[data2[var].apply(x23)]
            return data2

    def test_point_chart(self, point, check_var, indi_var):  # asuming no x, and point is given
        print('-test point chart')
        data = self.wether[self.cur_el]['t']
        data_indi = self.interp_data(data, point[indi_var], indi_var)
        if check_var in 'vhs':  # todo find val
            f, g = data_indi[check_var + 'f'][0], data_indi[check_var + 'g'][0]  # todo at interp
            dgf = g - f
            if 0 <= (point[check_var] - f) / dgf <= 1:
                ty = 't'
                # x = self.get_quality([f,g], point[check_var])
            else:  # not in between thus < g = <f
                si = point[check_var] - g
                si2 = si * dgf
                if si2 > 0:
                    ty = 'superheated'
                else:
                    ty = 'compressed'  # todo if no compressed
        elif data_indi[check_var][0] == point[check_var]:
            ty = 't'
        else:
            ty = self.test_point_chart(point, 's', indi_var)
        return ty

    def point_to_x(self, point, point2, check_var, indi_var):
        print('-point 2 ex')

        def p2(tix, rix):
            print('p2')
            data_parts = self.wether[self.cur_el][tix]
            data_partt = self.interp_data(data_parts, point[indi_var], indi_var, tix)
            for r in rix:
                print('rix: ', r)
                point_lst.append(self.interp_data(data_partt, r, check_var))

        ty = self.test_point_chart(point, check_var, indi_var)
        ty2 = self.test_point_chart(point2, check_var, indi_var)
        data = self.wether[self.cur_el]['t']
        data_indi = self.interp_data(data, point[indi_var], indi_var)
        point_lst = []
        f, g = data_indi[check_var + 'f'][0], data_indi[check_var + 'g'][0]
        vp = point[check_var]
        vp2 = point2[check_var]

        if ty == ty2:
            if ty == 't':
                x = self.get_quality([f, g], vp)
                x2 = self.get_quality([f, g], vp2)
                ra = np.linspace(x, x2)  # todo x,1
                for ri in ra:
                    point_lst.append(self.at_quality([f, g], ri))
            else:
                ra = linspace(vp, vp2)
                p2(ty, ra)


        else:
            i = [(vp, ty), (vp2, ty2)]
            for n in range(2):
                if i[n][1] == 't':
                    x = self.get_quality([f, g], vp)
                    if i[(n + 1) % 2][1] == 'superheated':
                        ra = np.linspace(x, g)  # todo x,1
                    else:
                        ra = np.linspace(f, g)  # todo x,1
                    if n == 0:
                        ra = ra[::-1]
                    for ri in ra:
                        point_lst.append(self.at_quality([f, g], ri))
                else:
                    if i[n][1] == 'superheated':
                        ti = g
                    else:
                        ti = f
                    ra = np.linspace(i[n][0], ti)
                    p2(i[n][1], ra)
        return point_lst

    def load_data(self):
        print('-load ex')

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

    def run_cmd(self, i, j=None):
        print('-rcm')
        print(f'run cmd i,j: ({i}, {j})')
        if i == self.m_d_s_n:
            print('-input')
            self.reset_inputs(j)
        elif i == self.s_d_s_n:
            print('-output')
            self.reset_outputs(j)

    def reset_inputs(self, ii=None):
        print(f'\n\n------------\n--------------\nrun reset iputs ii: ({ii})')
        if ii is not None:
            print('reset curel')
            self.cur_el = ii
        print(f'second data reset: cur el = ', self.cur_el)
        self.plot.reset_el()
        self.second_data_selector.reset_properties()

    def reset_outputs(self, jj=None):

        print(f'\n------------\nrun reset outputs jj: ({jj})')
        if jj is not None:
            # print('reset curty')
            self.cur_ty = jj
        # print(f'cur data reset: cur ty=', self.cur_ty)
        self.active_varable = []
        self.current_data = self.wether[self.cur_el][self.cur_ty]
        # print(f'var reset')
        self.varis.reset_data_columns()
        # todo current data?

        self.reset_table()

    def proces_lines_sat(self, points):
        print('-proces_lines_sat')
        num_p = len(points)
        if num_p > 1:
            for i in self.jk:
                da = []
                da_y = []
                for ij in range(num_p):  # todo invalid, process
                    # todo type
                    proces = self.proces[1]
                    pm = self.proscess_mean[proces]
                    d = self.point_to_x(points[ij][0], points[(ij + 1) % num_p][0], i[1], pm)
                    for di in d:
                        da.append(di[i[1]])
                        da_y.append(di[i[0]])

                self.ps[i].setData(da, da_y)  # todo next point on f or g or other

    def con_v(self, i, const_val, x=0, data=None):
        print('-con v')
        if data is None:
            data = self.current_data
        # di = self.wether[self.cur_el][self.cur_ty]  # todo any two data peices
        ind_1 = data[i][1:] > const_val
        ind_1 = ind_1.values
        index_2 = [ind_1[i] != ind_1[i + 1] for i in range(ind_1.size - 1)]

        ind_3 = index_2.index(True)

        lo = data.loc[ind_3:ind_3 + 1]

        dx = np.array(lo[i])
        p2 = {}
        for ii in data.columns:
            if ii != i and ii != 'x':
                if ii.replace('g', '').replace('f', '') in self.vari_ls and self.cur_ty in 'tp':
                    v_r = self.at_quality([self.interp(dx, data[ii + f], const_val) for f in 'fg'], x)
                    t_r = round(v_r, 4)
                    p2[ii.replace('g', '').replace('f', '')] = t_r
                else:
                    dy = np.array(lo[ii])
                    v_ret = self.interp(dx, dy, const_val)
                    t = round(v_ret, 4)
                    p2[ii] = t
                    print(f'ii,i ({ii},{i} set to {t}')
        return pd.DataFrame(p2)
        # self.set_point()

    def _set_table(self):
        print('-set table')
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
        print('-reset table')
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
        print('-tab_s')
        col_n = self.table.currentColumn()
        r_n = self.table.currentItem().text()

        col = list(self.current_data.columns)[col_n]
        self.varis.but[col].setText(r_n)
        self.reset_v(col, r_n)

    def interp(self, dx, dy, x):
        # print('-interp')
        if len(dx) < 2:
            return dy[0]
        else:
            return self.at_quality(dy, self.get_quality(dx, x))

    # todo two var for superheated,  plot on same t,p the s v curves off the charts
    def point_n(self, n):
        return float(self.varis.but[n].text())

    def set_point(self):
        print('-set point')
        self.plot_v.plot(self.point_n('v'), self.point_n('p'))
        self.plot_s.plot(self.point_n('s'), self.point_n('t'))
        pass

    def get_quality(self, dx, x):
        # print('-getquality')
        d = np.diff(dx)[0]
        if d == 0:
            d = 1
        return (x - dx[0]) / d

    def at_quality(self, dy, x):
        return np.diff(dy)[0] * x + dy[0]

    def reset_vs(self, i, tex,d=None):
        print('testing other option')
        if isinstance(tex,str):
            val = float(tex)
        else:
            val = tex
        if d is None:
            d = self.current_data[1:]
        min_d = d.loc[d[i] <= val, i].max()
        max_d = d.loc[d[i] >= val, i].min()
        fin_tab = d[d[i].between(min_d, max_d)]
        dfs = list(tuple(fin_tab.groupby(i)))  # list of all elements of data
        df_n = {}
        if len(dfs) == 1:
            for ii in fin_tab.columns:
                df_n[ii] = []
                r_ret = np.array(dfs[0][1][ii])

                for va in r_ret:
                    t = va
                    df_n[ii].append(t)
                    # self.varis.but[ii].setText(t)
                    print(f'data: (ii,i): ({ii},{i}) set to {t}')

        else:
            for n in range(len(dfs) - 1):
                dx1 = np.array(dfs[n][1][i]).flatten()
                dx2 = np.array(dfs[n+1][1][i]).flatten()
                dx = np.array([dx1[0], dx2[0]])
                len_t = len(dx1)
                for ii in fin_tab.columns:
                    if ii not in df_n:
                        df_n[ii] = []
                    if ii != i:
                        dv1 = np.array(dfs[n][1][ii]).flatten()
                        dv2 = np.array(dfs[n + 1][1][ii]).flatten()
                        for iii in range(len_t):
                            dy = np.array([dv1[iii], dv2[iii]]).flatten()
                            v_ret = self.interp(dx, dy, val)
                            df_n[ii].append(v_ret)
                        # self.varis.but[ii].setText(t)

                    else:
                        v_ret = [val]*len_t
                        # t = str((round(v_ret, 4)))
                        print(f'data: (ii,i): ({ii},{i}) set to {v_ret}')
                        df_n[ii].extend(v_ret)

        return pd.DataFrame(df_n)

    def d_set(self,dat, i, xi=None):
        for ii in dat.columns:

            if ii != i and ii != 'x':
                v_ret = np.array(dat[ii])[0]
                t = str((round(v_ret, 4)))
                self.varis.but[ii].setText(t)
                print(f'ii,i ({ii},{i} set to {t}')
        if xi is not None:
            t = str((round(xi, 4)))
            self.varis.but['x'].setText(t)
            for ij in self.vari_ls:
                vr = np.array([dat[ij+'f'], dat[ij+'g']]).flatten()
                v_ret = self.at_quality(vr, xi)
                t = str((round(v_ret, 4)))
                self.varis.but[ij].setText(t)

    def reset_v(self, i, tex):
        print('-reset val normal')
        print('reset vals: i=', i)  # todo x
        val = float(tex)
        print(f'---active v: {self.active_varable}---')
        if self.cur_ty in ['superheated', 'compressed']:

            if len(self.active_varable) ==0 or (len(self.active_varable) ==1 and i in self.active_varable):
                da = self.reset_vs(i, tex)
                self.d_set(da,i)
                self.active_varable.append(i)
            # self.set_point()
            else:
                da = self.reset_vs(self.active_varable[0], self.varis.but[self.active_varable[0]].text())
                da2 = self.reset_vs(i,tex,da)
                self.d_set(da2, i)
        elif self.cur_ty in 'tp':
            if i in self.vari_ls or i =='x':
                if len(self.active_varable) == 0 or (len(self.active_varable) == 1 and i in self.active_varable):
                    self.active_varable.append(i)
                elif self.active_varable[0] in self.vari_ls or self.active_varable[0]== 'x':
                    da = self.current_data[1:]
                    if i == 'x':

                        dd = self.varis.but[self.active_varable[0]].text()
                        ddx = val
                    else:
                        ddx = self.varis.but[self.active_varable[0]].text()
                        dd = val
                    for r in range(da.shape[0]):
                        ddy = np.array(da.loc[r, [i+'f',i+'g']])

                        out_v = self.at_quality(ddy,ddx)
                        if np.abs((out_v-dd)/dd) <= 0.01:
                            da2 = da.loc[r]
                            self.d_set(da2,i,ddx)
                            break
                    # find place where at quality x of fg = n, todo add error and precision
                    pass
                else:
                    da = self.reset_vs(self.active_varable[0], self.varis.but[self.active_varable[0]].text())
                    if i == 'x':
                        xii = val
                    else:
                        vr = np.array([da[i + 'f'], da[i + 'g']]).flatten()
                        xii = self.get_quality(vr, val)
                    self.d_set(da, i, xii)

            else:
                if len(self.active_varable) > 0 :
                    if self.active_varable[0] in self.vari_ls or self.active_varable[0] == 'x':
                        da = self.reset_vs(i, tex)

                        if self.active_varable[0] == 'x':
                            xii = val
                        else:
                            vr = np.array([da[self.active_varable[0] + 'f'], da[self.active_varable[0] + 'g']]).flatten()
                            xii = self.get_quality(vr, self.varis.but[self.active_varable[0]].text())
                        self.d_set(da, i, xii)
                    else:
                        da = self.reset_vs(i, tex)
                        self.d_set(da, i, 0)
                        self.active_varable = [i]

                else:
                    da = self.reset_vs(i, tex)
                    self.d_set(da, i, 0)
                    self.active_varable = [i]

    def _update_set(self):
        print('loading_all')
        print('-load user')
        self.setting_keys_combo = {self.m_d_s_n: 'water', self.s_d_s_n: 'p'}  # todo current windows, current el

        self.font_ty_default = {}

        self.settings.beginGroup('combo')

        val = self.settings.value(self.m_d_s_n, 'water')
        self.main_data_selector.setCurrentText(val)
        self.cur_el = val
        self.cur_ty = list(self.wether[self.cur_el].keys())[0]
        self.second_data_selector.reset_properties()
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
        print('-save user')
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
        print('-close window')
        self.user_settings()
        super().closeEvent(event)


# class DataF:
#     def __init__(self,df):
#         self.df = df
#         self.varis = 'hsv'
#         # super().__init__()
#     def __getitem__(self, item):
#         def reset_v(self, i, tex):
#             def check_v(iiii):
#                 v_r = self.at_quality([float(self.varis.but[iiii + f].text()) for f in 'fg'], self.qual)
#                 t_r = str((round(v_r, 4)))
#                 self.varis.but[iiii].setText(t_r)
#
#             print('reset vals: i=', i)  # todo x
#
#             val = float(tex)
#             if i == 'x':  # user input of independant and x gives y or y and x rare test
#                 self.qual = val
#                 for ii in self.vari_ls:
#                     check_v(ii)
#
#             elif i in self.vari_ls and self.cur_ty in 'tp':
#                 he = [float(self.varis.but[i + f].text()) for f in 'fg']  # todo empty
#                 if min(he) <= val <= max(he):
#                     self.qual = self.get_quality(he, val)
#                     for ii in self.vari_ls:
#                         if ii != i:
#                             check_v(ii)
#                 else:
#                     for ii in self.varis.but.keys():
#                         if ii != i:
#                             self.varis.but[ii].setText('0')
#             else:
#
#                 # di = self.wether[self.cur_el][self.cur_ty]  # todo any two data peices
#                 ind_1 = self.current_data[i][1:] > val
#                 ind_1 = ind_1.values
#                 index_2 = [ind_1[i] != ind_1[i + 1] for i in range(ind_1.size - 1)]
#
#                 ind_3 = index_2.index(True)
#
#                 lo = self.current_data.loc[ind_3:ind_3 + 1]
#                 # if self.cur_ty not in 'tp':
#                 #     if i == list(self.varis.but.keys())[0]:
#
#                 dx = np.array(lo[i])
#
#                 for ii in self.varis.but.keys():
#
#                     if ii != i and ii != 'x':
#                         if ii in self.vari_ls and self.cur_ty in 'tp':
#                             check_v(ii)
#                         else:
#                             dy = np.array(lo[ii])
#                             v_ret = self.interp(dx, dy, val)
#                             t = str((round(v_ret, 4)))
#                             self.varis.but[ii].setText(t)
#                             print(f'ii,i ({ii},{i} set to {t}')
#             # self.set_point()
#             # todo plot proces on charts
#         if isinstance(item, (int, slice)):
#             return self.__class__(self.value[item])
#         return [self.value[i] for i in item]
class SuperCombo(QComboBox):
    def __init__(self, name, par, orient_v=True, vals=None, show_lab=True, run=True):
        super().__init__()
        print('-init_super ', name)
        self.par = par
        self.orient_v = orient_v
        self.show_lab = show_lab
        self.name = name
        self.wig = QWidget()
        self.lab = QLabel(self.name)
        self._layout_set()
        print('vals')
        if vals is not None:
            self.addItems(vals)

        if run:
            print('run')

    def init_op(self):
        print('-initop')
        self.currentTextChanged.connect(self.rc)

    def rc(self):
        print('-rc')
        x = self.currentText()
        self.currentTextChanged.disconnect(self.rc)
        if x is not None and x != '':
            print('rc current text')
            self.par.run_cmd(self.name, x)
            self.currentTextChanged.connect(self.rc)

    # noinspection PyArgumentList
    def _layout_set(self):
        print('-layoutset')
        print(f'super combo: {self.name}, layout set')
        if self.orient_v:
            self.layout = QVBoxLayout()
        else:
            self.layout = QHBoxLayout()
        self.layout.addWidget(self)
        self.layout.addWidget(self.lab)
        self.wig.setLayout(self.layout)

    def reset_properties(self):
        print('-reset prop')
        try:
            self.currentTextChanged.disconnect(self.rc)
            print('disconect\n')
        except TypeError:
            print('error')

        self.clear()

        self.addItems(list(self.par.wether[self.par.cur_el].keys()))

        if self.par.cur_ty in self.par.wether:
            self.setCurrentText(self.par.cur_ty)

        self.currentTextChanged.connect(self.rc)


class SuperText(QWidget):
    def __init__(self, name, par, orient_v=True, show_lab=True):
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

    def removeButtons(self):
        print('-rm button')
        for cnt in reversed(range(self.layout2.count())):
            # takeAt does both the jobs of itemAt and removeWidget
            # namely it removes an item and returns it
            widget = self.layout2.takeAt(cnt).widget()

            if widget is not None:
                # widget will be None if the item is a layout
                widget.deleteLater()

    def reset_data_columns(self):
        print('____________________\n__________________\nreset data')
        vals = list(self.par.current_data.columns)
        if self.par.cur_ty in 'tp':
            vals.extend(['v', 'h', 's', 'x'])  # todo always?

        self.removeButtons()
        self.lab_b = {}
        self.but = {}
        print('____________________\nend rm\n')
        # for ij in vm:
        #     del self.lab_b[ij]
        #     del self.but[ij]

        n = 0
        mi = 0
        for i in vals:
            # if i not in self.but.keys():

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
            self.layout2.addWidget(j, m + 1, ni)  # todo add lable for each
        self.update()
        print('end data add\n____________________\n')

    def run_cm(self, i):
        print('-run cm')
        tex = self.but[i].text()
        print(f'________________\ntex={tex}\n-------------------')
        # print(tex)
        self.par.reset_v(i, tex)


class point_ls(QListWidget):
    def __init__(self, par):
        super().__init__()
        # self.setWindowTitle('QMainWindow')
        self.full_c_v = []
        self.par = par
        self.curr_type = 'Const V'

        self._create_tools()
        # self.cam = cam(self, **self.va)
        self.list_index = 0
        # self.cen = aniWig(self)
        # self.insert_list_wig_vals('+')
        # self._set_p()

    def _create_tools(self):

        def add_push_button(i):
            print('-add push but')
            if i == '=':
                kk = '+'
            else:
                kk = i
            print('action I')
            j = QPushButton(kk)
            k = QAction(i)
            k.triggered.connect(partial(self.insert_list_wig_vals, i))
            j.clicked.connect(partial(self.insert_list_wig_vals, i))
            self.list_v[i] = j

            self.action_list[i] = k
            return j, k

        print('-ceate tool')
        self.tool_dock = QDockWidget('list widget')
        # add rem move up move down, single drag, selct, doble edit
        self.tools = QWidget()
        self.tool_dock.setWidget(self.tools)

        self.par.addDockWidget(Qt.LeftDockWidgetArea, self.tool_dock)
        self.tl = QVBoxLayout()
        self.tool_layout = QGridLayout()
        self.pm_l = QHBoxLayout()
        self.lr_l = QVBoxLayout()
        self.tools.setLayout(self.tool_layout)
        self.list_v = {}
        self.action_list = {}

        self.itemClicked.connect(self.item_swap)

        for add_comand in '=-':
            ji, ki = add_push_button(add_comand)
            self.pm_l.addWidget(ji)
            ki.setShortcut(f"ctrl+{add_comand}")

        for add_comand in ['Up', 'Down']:
            ji, ki = add_push_button(add_comand)
            self.lr_l.addWidget(ji)
            ki.setShortcut(add_comand)

        self.tool_layout.addWidget(self, 0, 0, 1, 2)
        self.tool_layout.addLayout(self.lr_l, 1, 0)
        self.tool_layout.addLayout(self.pm_l, 1, 1)
        self.in_p = {}

    def item_swap(self, item):
        print('-item swap')
        self.setCurrentItem(item)
        self.list_index = self.ls_w.currentRow()

    def add_motion(self, data, replace=False):
        print('-add motion')
        print('replace: ', replace)
        kkkk = len(self.full_c_v)
        ty = self.par.cur_ty
        if replace:

            self.full_c_v[self.list_index] = (data, ty)
            # self.item_swap()
            ite = self.item(self.list_index)
            te = ite.text()
            te = te.replace(' ', '').split(':')[-1]
            ite.setText(f'{ty} Point: {te}')
        else:
            self.full_c_v.insert(self.list_index + 1, (data, ty))
            self.insertItem(self.list_index, f'{ty} Point: {kkkk}')
        self.par.plot.res(self.full_c_v)
        self.update()

    def insert_list_wig_vals(self, input_command):
        print('-insert_list_wig_vals')
        print('I', input_command)
        self.list_index = self.currentRow()
        if input_command == '+' or input_command == '=':  # todo add to menu
            # remove

            dic_d = {}
            for k, v in self.par.varis.but.items():
                print(f'k,v: ({k}:{v.text()})')
                dic_d[k] = float(v.text())
            self.add_motion(dic_d)  # todo maybe listitem? and check change motion
            # self.mo.insert(ind)

            # remove dict
        elif input_command == '-':
            # remove
            self.removeItem(self.list_index)
            self.full_c_v.remove(self.list_index)
            # remove dict
            # todo rem from cam
            pass
        elif input_command == 'Up':
            print('up')
            self.ls_w.currentItem.moveUp()
        else:
            print('down')
            self.ls_w.currentItem.moveDown()


if __name__ == "__main__":
    print('Running Chem E Solve')
    app = QApplication(sys.argv)
    audio_app = Window()
    audio_app.show()
    sys.exit(app.exec_())
