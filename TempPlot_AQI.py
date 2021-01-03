import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def avg_data():
     yearavg = {}
     for year in range(2013, 2019):
         temp_i = 0
         average = []
         for rows in pd.read_csv("Data/AQI/aqi{}.csv".format(year), chunksize=24):
             add_var = 0
             avg = 0.0
             data = []
             df = pd.DataFrame(data=rows)
             for index, row in df.iterrows():
                 data.append(row['PM2.5'])
             for i in data:
                 if type(i) is float or type(i) is int:
                     add_var = add_var + i
                 elif type(i) is str:
                     if i != 'NoData' and i != 'PwrFail' and i != '---' and i != 'InVld':
                         temp = float(i)
                         add_var = add_var + temp
             avg = add_var / 24
             temp_i = temp_i + 1

             average.append(round(avg,3))
             yearavg[year] = average
     print(yearavg)
     return yearavg


if __name__ == '__main__':
   year_dict = avg_data()
   print(year_dict)
   lst2013 = year_dict.get(2013)
   lst2014 = year_dict.get(2014)
   lst2015 = year_dict.get(2015)
   lst2016 = year_dict.get(2016)
   plt.plot(range(0, 365), lst2013, label = '2013 Year')
   plt.plot(range(0, 364), lst2014, label = '2014 Year')
   plt.plot(range(0, 365), lst2015, label = '2015 Year')
   plt.plot(range(0, 365), lst2016, label = '2016 Year')
   plt.xlabel('Day')
   plt.ylabel('PM 2.5')
   plt.legend(loc = 'upper right')
   plt.show()

##Another method
# from glob import glob
# def avg_data():
#     filenames = glob('Data/AQI/*.csv')
#     dataframes = [f for f in filenames]
#     avg_list_for_allYears = []
#     for f in filenames:
#         temp_i = 0
#         average = []
#         for rows in pd.read_csv(f, chunksize = 24):
#             add_var = 0
#             avg = 0.0
#             data = []
#             df =pd.DataFrame(data = rows)
#             for index, row in df.iterrows():
#                 data.append(row['PM2.5'])
#             for i in data:
#                 if type(i) is float or type(i) is int:
#                     add_var = add_var + i
#                 elif type(i) is str:
#                     if i != 'NoData' and i != 'PwrFail' and i != '---' and i != 'InVld':
#                         temp = float(i)
#                         add_var = add_var + temp
#             avg = add_var / 24
#             temp_i = temp_i + 1
#             average.append(round(avg,3))
#             avg_list_for_allYears.append(average)
#         return avg_list_for_allYears
#
# if __name__ == "__main__":
#     lst_all_years = avg_data()
#     #print(lst_all_years)
#
#     a = [2013, 2014, 2015, 2016, 2017, 2018]
#     for i in range(len(a)):
#         for j in lst_all_years:
#             size = "Days in year {} : {}".format(a[i], len(j))
#         print(size)




















