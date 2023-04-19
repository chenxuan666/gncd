import pandas as pd
import scipy
from pandas import Series
from pandas.testing import assert_frame_equal
import numpy as np

from scipy import ndimage

crops = ['maize','wheat','soy']#, 'millet', 'sorghum', 'rice', 'wheat', 'cassava', 'bean']

def calculate_global_total_yield_anomalies():
    for crop in crops:
        methods = ['Gau','Smooth9','Smooth5']
        for method in methods:
            df = pd.read_csv('data/crop/'+crop+'_world_total_yield.csv')
            df = df[df['Element'] == 'Yield']
            if method == 'Gau':
                df['yldGauExp'] = ''
                df['percent_yield_anomaly'] = ''
                # calculate the low-frequency expected yield from the 1-d yield time series (“cntYield.values”)
                df['yldGauExp'] = ndimage.filters.gaussian_filter1d(df.Value.values, 3)
                # subtract the low-frequency mean (“expected yield”) from the absolute yield and divide by the expected yield (e.g. percent yield anomaly = (yield – expected yield) / expected yield)
                df['percent_yield_anomaly'] = (df.Value.values - df.yldGauExp) / df.yldGauExp
            if method == 'Smooth5':
                df['ma5'] = ''
                df['crop_yield_anomaly'] = ''
                df['percent_yield_anomaly'] = ''
                # df['percent_lower_than_-5%'] = ''
                # df['percent_lower_than_-10%'] = ''
                # df['percent_bottom_quartile'] = ''
                poor_year_list = []
                for country in df['Area'].drop_duplicates():
                    df1 = df.loc[df['Area'] == country]
                    if len(df1['Value']) >= 5:
                        dfm5 = running_mean(df1['Value'].to_list(), 5)
                        dfm5.insert(0, 0)
                        dfm5.insert(1, 0)
                        dfm5.insert(len(dfm5), 0)
                        dfm5.insert(len(dfm5), 0)
                        # print(country)
                        df1['ma5'] = dfm5
                    else:
                        df1['ma5'] = df1['Value']
                    # df = df.loc[df['Area'] == country]
                    # calculate 5-year running mean
                    # df['ma5'] = df['Value'].rolling(window=5).mean()  # Value.ewm(span = 10, adjust = False).mean()#
                    # df.fillna(0, inplace=True)
                    #   replace ma5 with value when ma5 is nan
                    df1['ma5'][df1['ma5'] == 0] = df1[df1['ma5'] == 0]['Value']
                    df1['crop_yield_anomaly'] = df1['Value'] - df1['ma5']
                    df1['percent_yield_anomaly'] = df1['crop_yield_anomaly'] / df1['ma5'] * 100
                    # Identify every year when the percent crop yield anomaly is s in the bottom 1/4 of all yield anomalies
                    #     df1.sort_values(by='percent_yield_anomaly', inplace=True, ascending=True)
                    #     df1.loc[df1.head(round(0.25 * len(df1))).index, 'percent_bottom_quartile'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -5% for each country
                    #     df1.loc[(df1.percent_yield_anomaly < -5), 'percent_lower_than_-5%'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -10% for each country
                    #     df1.loc[(df1.percent_yield_anomaly < -10), 'percent_lower_than_-10%'] = 'poor'
                    df1.sort_values(by='Year', inplace=True, ascending=True)
                    # rewrite df with df

                    # print(important_countries_list)
                    # Identify when there are years with at least 4 important countries that have poor yields
                    df1.crop_yield_anomaly.fillna(0, inplace=True)
                    df[df['Area'] == country] = df1[df1['Area'] == country]
            if method == 'Smooth9':
                df['ma9'] = ''
                df['crop_yield_anomaly'] = ''
                df['percent_yield_anomaly'] = ''
                # df['percent_lower_than_-5%'] = ''
                # df['percent_lower_than_-10%'] = ''
                # df['percent_bottom_quartile'] = ''
                poor_year_list = []
                for country in df['Area'].drop_duplicates():
                    df1 = df.loc[df['Area'] == country]
                    if len(df1['Value']) >= 9:
                        dfm5 = running_mean(df1['Value'].to_list(), 9)
                        dfm5.insert(0, 0)
                        dfm5.insert(0, 0)
                        dfm5.insert(0, 0)
                        dfm5.insert(0, 0)
                        dfm5.insert(len(dfm5), 0)
                        dfm5.insert(len(dfm5), 0)
                        dfm5.insert(len(dfm5), 0)
                        dfm5.insert(len(dfm5), 0)
                        # print(country)
                        df1['ma9'] = dfm5
                    else:
                        df1['ma9'] = df1['Value']
                    #   replace ma5 with value when ma5 is nan
                    df1['ma9'][df1['ma9'] == 0] = df1[df1['ma9'] == 0]['Value']
                    df1['crop_yield_anomaly'] = df1['Value'] - df1['ma9']
                    df1['percent_yield_anomaly'] = df1['crop_yield_anomaly'] / df1['ma9'] * 100
                    # Identify every year when the percent crop yield anomaly is s in the bottom 1/4 of all yield anomalies
                    #     df1.sort_values(by='percent_yield_anomaly', inplace=True, ascending=True)
                    #     df1.loc[df1.head(round(0.25 * len(df1))).index, 'percent_bottom_quartile'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -5% for each country
                    #     df1.loc[(df1.percent_yield_anomaly < -5), 'percent_lower_than_-5%'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -10% for each country
                    #     df1.loc[(df1.percent_yield_anomaly < -10), 'percent_lower_than_-10%'] = 'poor'
                    df1.sort_values(by='Year', inplace=True, ascending=True)
                    # rewrite df with df

                    # print(important_countries_list)
                    # Identify when there are years with at least 4 important countries that have poor yields
                    df1.crop_yield_anomaly.fillna(0, inplace=True)
                    df[df['Area'] == country] = df1[df1['Area'] == country]
            ds1 = pd.read_csv('table/number of countries/' + crop + '0.5'+method+'_number_every_year_important_countries.csv')
            ds2 = pd.read_csv('table/number of countries/' + crop + '0.55'+method+'_number_every_year_important_countries.csv')
            ds3=pd.read_csv('table/number of countries/' + crop + '0.6'+method+'_number_every_year_important_countries.csv')
            ds4 = pd.read_csv('table/number of countries/' + crop + '0.65'+method+'_number_every_year_important_countries.csv')
            ds5 = pd.read_csv('table/number of countries/' + crop + '0.7'+method+'_number_every_year_important_countries.csv')
            ds6 = pd.read_csv('table/number of countries/' + crop + '0.75'+method+'_number_every_year_important_countries.csv')
            ds7 = pd.read_csv('table/number of countries/' + crop + '0.8'+method+'_number_every_year_important_countries.csv')
            if crop == 'soy':
                ds8 = pd.read_csv('table/number of countries/' + crop + '0.85' + method + '_number_every_year_important_countries.csv')
                ds9 = pd.read_csv('table/number of countries/' + crop + '0.9' + method + '_number_every_year_important_countries.csv')
            df = df.drop(df.percent_yield_anomaly[df.percent_yield_anomaly == 0].index)
            df.reset_index()
            y = df['percent_yield_anomaly'].to_list()


            if crop == 'soy':
                dslist=[ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9]
            else:
                dslist = [ds1, ds2, ds3, ds4, ds5, ds6, ds7]
            # calculate pearsons rank correlation
            if crop == 'soy':
                pearson_corr=pd.DataFrame(columns=['50','55','60','65','70','75','80','85','90'])
            pearson_corr = pd.DataFrame(columns=['50', '55', '60', '65', '70', '75', '80'])
            for ds in dslist:
                x1 = scipy.stats.stats.rankdata(ds['number_of_important_countries_blew_-5%'].to_list())
                x2 = scipy.stats.stats.rankdata(ds['number_of_important_countries_blew_-10%'].to_list())
                x3 = scipy.stats.stats.rankdata(ds['number_of_important_countries_bottom_quartile'].to_list())
                corr1 = scipy.stats.stats.spearmanr(x1,y)
                corr2 = scipy.stats.stats.spearmanr(x2,y)
                corr3 =scipy.stats.stats.spearmanr(x3,y)
                corrlist=[corr1[0],corr2[0],corr3[0]]
                if ds.equals(dslist[0]):
                    pearson_corr['50']=corrlist
                if ds.equals(dslist[1]):
                    pearson_corr['55'] = corrlist
                if ds.equals(dslist[2]):
                    pearson_corr['60'] = corrlist
                if ds.equals(dslist[3]):
                    pearson_corr['65'] = corrlist
                if ds.equals(dslist[4]):
                    pearson_corr['70'] = corrlist
                if ds.equals(dslist[5]):
                    pearson_corr['75'] = corrlist
                if ds.equals(dslist[6]):
                    pearson_corr['80'] = corrlist
                if crop == 'soy':
                    if ds.equals(dslist[7]):
                        pearson_corr['85'] = corrlist
                    if ds.equals(dslist[8]):
                        pearson_corr['90'] = corrlist
            if method == 'Gau':
                pearson_corr.to_csv('table/spearman_corr/' + crop + 'yldAnomGau spearman_corr_update.csv')
            if method == 'Smooth5':
                pearson_corr.to_csv('table/spearman_corr/' + crop + 'yldAnomSmooth5 spearman_corr_update.csv')
            if method == 'Smooth9':
                pearson_corr.to_csv('table/spearman_corr/' + crop + 'yldAnomSmooth9 spearman_corr_update.csv')


def running_mean(data,w):
    res = []
    for i in range(len(data)-w+1):
        ave=0
        for j in range(w):
            ave += data[i+j]
        ave /= w
        res.append(ave)
    return res

calculate_global_total_yield_anomalies()