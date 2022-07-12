import pandas as pd
import numpy as np
import datetime
import calendar
import matplotlib.pyplot as plt

file = 'WeatherData.xlsx'
data = pd.read_excel(file)

# todo plot n year vaerage per day the
# last n yers
# roling average
# per station
# single
amm = ['Mean Temperature (C)', 'Maximum Temperature (C)', 'Minimum Temperature (C)']


# per day of week per day of moth
def ave_d(d):
    amm_d = {}
    for i in amm:
        amm_d[i] = np.mean(d[i])
    return amm_d


def grab_day(day, d, full=True):
    # date = day easyer
    #
    day_txt = 'Date' if full else 'Day'
    return ave_d(d.loc[d[day_txt] == day])

def grab_date(day, d):
    return ave_d(d.loc[lambda ddd: ddd['Date'].split('-',1)[1] == day])

def grab_month(m, d):
    return d.loc[d['Month'] == m]


def grab_years(start, stop, d=data):
    return d.loc[(d['Year'] >= start) & (d['Year'] <= stop)]


def cal_st(start, stop, d):
    # todo seprt
    station = {'all': grab_years(start, stop, d)}
    for i in ['yeg', 'n', 'm']:
        station[i] = grab_years(start, stop, data['Station'==i])


# def rolling_avg():
#     jj = np.array([])
#     for i in data['dates']:
#         jj[i['month']-1, day-1, year-2000] = ave_d(i)
#     for i in jj:
#         for j in i:
#             ave_d(j)
#     pass


def days_of_week(d):
    dd = []

    for i in range(7):
        da = d.loc[lambda s: datetime.weekday(s['date']) == i]  # todo year
        dd.append(ave_d(da))
    return dd


if __name__ == '__main__':
    print(grab_date('01-15',data))
    c = calendar.Calendar()
    day_ls = {}
    for mo in range(1,13):
        mo_d = grab_month(mo,data)
        for days in c.itermonthdays(2000, mo):
            if days != 0:
                st = str(mo)+"-"+str(days)
                day_ls[st] = grab_day(days, mo_d,False)

    plt.polar([x['Mean Temperature (C)'] for x in day_ls.values()])
    # k_ls = [x for n, x in enumerate(day_ls.keys()) if n %30==0]
    # plt.xticks(np.linspace(0,len(day_ls.keys()),len(k_ls)),k_ls)
    plt.show()
