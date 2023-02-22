from pandas import Series
from pandas.testing import assert_frame_equal
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import _pickle as cPickle
import os
import cartopy.io.shapereader as shpreader
import warnings
import heapq
from PIL import Image
from sklearn.linear_model import LinearRegression

from scipy import ndimage

crops = ['wheat','maize','soy']#,''wheat,'millet', 'sorghum', 'rice','cassava', 'bean']


def calculate_production():
    for crop in crops:
        df = pd.read_csv('data/crop/FAO_' + crop + '_1949_2020.csv')
        df = df[(df.year >= 1961) & (df.year <= 2020)]
        df = df.reindex(columns=['year', 'country', 'Production', 'yldAnomGau', 'yldAnomSmooth5', 'yldAnomSmooth9'])
        
        # if crop=='wheat':
        #     df=df[~(df['country']=='China')]
        #     df = df[~(df['country'] == 'India')]
            
        df.fillna(0, inplace=True)
        df['crop'] = crop

        clos1 = ['crop_type', 'country', 'average_production', 'percent']
        percent_of_important_countries = pd.DataFrame(columns=clos1)
        df.reset_index()
        ds = df.loc[(df['year'] >= 2015) & (df['year'] <= 2020)]
        for country in ds['country'].drop_duplicates():
            # calculate average total national production over the period 2015-2020
            average_total_production = ds[ds['country'] == country].Production.sum() / len(ds[ds['country'] == country])
            percent_of_important_countries.loc[len(percent_of_important_countries)] = [crop, country,
                                                                                       average_total_production,
                                                                                       0]  # ,average_total_production/total_global_production]
            # calculate average total global production
        average_total_global_production = percent_of_important_countries.average_production.sum()

        for country in percent_of_important_countries.country:
            #  calculate the percent of total global crop production produced by each country
            percent_of_important_countries.loc[
                (percent_of_important_countries.country == country), 'percent'] = float(
                percent_of_important_countries[
                    percent_of_important_countries.country == country].average_production / average_total_global_production)
            # sorted by percent
        percent_of_important_countries.sort_values(by='percent', ascending=False, inplace=True)
        percent_of_important_countries.reset_index()

        # Come up with our list of “important countries”
        if True:
            important_countries_list_after1990 = []
            total_percent = 0
            flag=0.80
            if crop == 'maize':
                flag = 0.75
            elif crop == 'wheat':
                flag = 0.75
            elif crop == 'soy':
                flag = 0.85
            for country in percent_of_important_countries.country:

                total_percent = total_percent + float(
                    percent_of_important_countries[percent_of_important_countries['country'] == country].percent)
                important_countries_list_after1990.append(country)

                if total_percent > flag:
                    break

        percent_of_important_countries['percent'] = percent_of_important_countries['percent'].apply(
            lambda x: format(x, '.4%'))
        percent_of_important_countries.to_csv('table/sorted by percent/' + crop + ' percent of important countries.csv')

        # create important countries columns
        important_countries_list = important_countries_list_after1990
        important_countries_list_drop_duplicates = []
        for i in important_countries_list:
            if i not in important_countries_list_drop_duplicates:
                important_countries_list_drop_duplicates.append(i)
        # create a dataframe with columns named important countries list
        # important_countries_experences_poor_year = pd.DataFrame([important_countries_list_drop_duplicates])
        # year_vals=df['year'].drop_duplicates()
        # important_countries_experences_poor_year=pd.DataFrame(index=year_vals,columns=[important_countries_list_drop_duplicates])
        # important_countries_experences_poor_year['year']=important_countries_experences_poor_year.index
        print(important_countries_list_drop_duplicates)

        # if crop=='wheat':
        #      important_countries_list_drop_duplicates.remove('China')
        #      important_countries_list_drop_duplicates.remove('India')
        #      important_countries_list_drop_duplicates.remove('United States of America')


        ds1 = df.reindex(columns=['crop', 'year', 'country', 'yldAnomSmooth5'])
        ds2 = df.reindex(columns=['crop', 'year', 'country', 'yldAnomGau'])
        ds3 = df.reindex(columns=['crop', 'year', 'country', 'yldAnomSmooth9'])
        dslistsmooth = [ds1, ds2, ds3]
        for df in dslistsmooth:

            df['percent_lower_than_-5%'] = ''
            df['percent_lower_than_-10%'] = ''
            df['percent_bottom_quartile'] = ''

            # calculate 5-year running mean and indentify the poor year
            for country in df['country'].drop_duplicates():
                df1 = df.loc[df['country'] == country]
                # Identify every year when the percent crop yield anomaly is s in the bottom 1/4 of all yield anomalies
                if df.equals(dslistsmooth[0]):
                    df1.sort_values(by='yldAnomSmooth5', inplace=True, ascending=True)
                    # Identify every year when the percent crop yield anomaly is lower than -5% for each country
                    df1.loc[(df1.yldAnomSmooth5 < -0.05), 'percent_lower_than_-5%'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -10% for each country
                    df1.loc[(df1.yldAnomSmooth5 < -0.10), 'percent_lower_than_-10%'] = 'poor'
                if df.equals(dslistsmooth[1]):
                    df1.sort_values(by='yldAnomGau', inplace=True, ascending=True)
                    # Identify every year when the percent crop yield anomaly is lower than -5% for each country
                    df1.loc[(df1.yldAnomGau < -0.05), 'percent_lower_than_-5%'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -10% for each country
                    df1.loc[(df1.yldAnomGau < -0.10), 'percent_lower_than_-10%'] = 'poor'
                if df.equals(dslistsmooth[2]):
                    df1.sort_values(by='yldAnomSmooth9', inplace=True, ascending=True)
                    # Identify every year when the percent crop yield anomaly is lower than -5% for each country
                    df1.loc[(df1.yldAnomSmooth9 < -0.05), 'percent_lower_than_-5%'] = 'poor'
                    # Identify every year when the percent crop yield anomaly is lower than -10% for each country
                    df1.loc[(df1.yldAnomSmooth9 < -0.10), 'percent_lower_than_-10%'] = 'poor'
                df1.loc[df1.head(round(0.25 * len(df1))).index, 'percent_bottom_quartile'] = 'poor'
                df1.sort_values(by='year', inplace=True, ascending=True)
                # rewrite df with df1
                df[df['country'] == country] = df1[df1['country'] == country]
            # print(important_countries_list)
            # Identify when there are years with at least 4 important countries that have poor yields
            # df.crop_yield_anomaly.fillna(0, inplace=True)
            # filling background_color for blew -5%
            # dfbottom=df
            # df['percent_yield_anomaly']
            ds_5 = df[df['percent_lower_than_-5%'] == 'poor']
            ds_10 = df[df['percent_lower_than_-10%'] == 'poor']
            ds_bottom_quartile = df[df['percent_bottom_quartile'] == 'poor']
            dslist = [ds_5, ds_10, ds_bottom_quartile]

            # clos0 = ['crop', 'poor_year', 'number', 'poor_year_important_countries_blew_-5%']
            # poor_year_important_countries = pd.DataFrame(columns=clos0)
            clos2 = ['crop', 'year', 'number_of_important_countries_blew_-5%',
                     'number_of_important_countries_blew_-10%',
                     'number_of_important_countries_bottom_quartile']
            every_year_important_countries = pd.DataFrame(columns=clos2)
            if df.equals(dslistsmooth[0]):
                every_year_important_countries['yldAnomSmooth5'] = ''
            if df.equals(dslistsmooth[1]):
                every_year_important_countries['yldAnomGau'] = ''
            if df.equals(dslistsmooth[2]):
                every_year_important_countries['yldAnomSmooth9'] = ''
                # important_countries_list = Series(important_countries_list)
            important_countries_list_after1990 = Series(important_countries_list_after1990)
            # important_countries_list_after1990 = Series(important_countries_list_after1990)

            # Time series of the number of important countries that meet our “poor year” threshold for each year
            for ds in dslist:
                # changes_of_important_countries=pd.DataFrame(columns=['year','country'])
                ds.sort_values(by='year', inplace=True)

                for year in ds.year.drop_duplicates():
                    intersection_important_countries = pd.Series(
                        list(set(ds[ds['year'] == year].country).intersection(important_countries_list_drop_duplicates)))

                    # for country in intersection_important_countries:
                    #     changes_of_important_countries.loc[len(changes_of_important_countries)]=[year,country]
                    if ds.equals(dslist[0]):
                        every_year_important_countries.loc[len(every_year_important_countries)] = [crop, year,
                                                                                                   len(intersection_important_countries),
                                                                                                   0, 0, 0]

                    elif ds.equals(dslist[1]):
                        every_year_important_countries.loc[every_year_important_countries[
                                                               'year'] == year, 'number_of_important_countries_blew_-10%'] = len(
                            intersection_important_countries)
                    else:
                        every_year_important_countries.loc[every_year_important_countries[
                                                               'year'] == year, 'number_of_important_countries_bottom_quartile'] = len(
                            intersection_important_countries)

                year_vals = df['year'].drop_duplicates()
                indexlist = year_vals.to_list()
                model_table = pd.DataFrame(index=indexlist, columns=[important_countries_list_drop_duplicates])
                for year in ds.year.drop_duplicates():
                    intersection_imporatnt_countries_experences_poor_year = pd.Series(
                        list(set(ds[ds['year'] == year].country).intersection(important_countries_list_drop_duplicates)))
                    if len(intersection_imporatnt_countries_experences_poor_year) > 0:
                        for country in intersection_imporatnt_countries_experences_poor_year:
                            model_table.loc[year, country] = 1
                    else:
                        model_table.loc[year, country] = 0
                    # important_countries_experences_poor_year.loc[year, 'number of important countries'] = df[(df['Area'] == country)&(df['Year']==year)].percent_yield_anomaly.values[0]
                model_table.fillna(0, inplace=True)
                # model_table.astype(int)
                define_method = ''
                if ds.equals(dslist[0]):
                    define_method = 'blew -5%'
                if ds.equals(dslist[1]):
                    define_method = 'blew -10%'
                if ds.equals(dslist[2]):
                    define_method = 'bottom quartile'
                # time_series_of_yield_anomalies(crop,define_method, important_countries_list_after1990, df,every_year_important_countries,model_table)
                #     poor_year_important_countries.loc[len(poor_year_important_countries)] = [crop, year, len(intersection_important_countries ),impotant_countries]

            # Create grid graphs for each definition of poor year (e.g. 5%, 10%, and bottom quartile) that shows which countries
            #  experience a poor year in which years.
            every_year_important_countries.replace(0,np.nan)
            # if df.equals(dslistsmooth[0]):
            #     every_year_important_countries.to_csv( 'table/number of countries/' + crop +str(flag)+ 'Smooth5_number_every_year_important_countries_with0.csv')
            # if df.equals(dslistsmooth[1]):
            #     every_year_important_countries.to_csv(
            #         'table/number of countries/' + crop + str(flag) + 'Gau_number_every_year_important_countries_with0.csv')
            # if df.equals(dslistsmooth[2]):
            #     every_year_important_countries.to_csv(
            #         'table/number of countries/' + crop + str(flag) + 'Smooth9_number_every_year_important_countries_with0.csv')
            pore='yield'
            # calculate_the_linear_trends(crop, every_year_important_countries,pore)
            # calculate_censored_percent_yield_anomaly_time_series(crop,important_countries_list_after1990,df)
            # combine_image(crop)
            # pooryears = calculate_the_poor_years(crop, every_year_important_countries)
            # print(pooryears)
            # if crop == 'wheat':
            #     pooryears.append(2007)
            # crop_yield_anomalies_spatial_map(crop, df, pooryears)

        #     year_vals = ds['Year'].drop_duplicates()
        #     indexlist=year_vals.to_list()
        #     # indexlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]
        #     important_countries_experences_poor_year = pd.DataFrame(index=indexlist,columns=[important_countries_list_drop_duplicates])
        #     model_table=pd.DataFrame(index=indexlist,columns=[important_countries_list_drop_duplicates])
        #     # important_countries_experences_poor_year.insert(loc=0, column='year', value=year_vals.to_list())
        #     # important_countries_experences_poor_year.loc[3000]
        #     for year in ds.Year.drop_duplicates():
        #         if year < 1992:
        #             intersection_imporatnt_countries_experences_poor_year = pd.Series(
        #                         list(set(ds[ds['Year'] == year].Area).intersection(important_countries_list_after1990)))
        #         # elif year >= 1992:
        #         #     intersection_imporatnt_countries_experences_poor_year = pd.Series(
        #         #                 list(set(ds[ds['Year'] == year].Area).intersection(important_countries_list_after1990)))
        #          # intersection_imporatnt_countries_experences_poor_year= pd.Series(
        #          #     list(set(ds[ds['Year'] == year].Area).intersection(important_countries_list_drop_duplicates)))
        #
        #         # filling grid plot with  yield anomaly percent
        #         for country in important_countries_list_drop_duplicates :
        #                  # for column in important_countries_experences_poor_year.columns:
        #                  #
        #                  # print(df[(df['Area'] == country)&(df['Year']==year)].percent_yield_anomaly.values[0])
        #                  if df[(df['Area'] == country)&(df['Year']==year)].percent_yield_anomaly.values :
        #                      important_countries_experences_poor_year.loc[year,country]=df[(df['Area'] == country)&(df['Year']==year)].percent_yield_anomaly.values
        #                  else:
        #                      important_countries_experences_poor_year.loc[year, country] =0
        #         important_countries_experences_poor_year.loc[year,'number of important countries']=len(intersection_imporatnt_countries_experences_poor_year)
        #
        #         if len(intersection_imporatnt_countries_experences_poor_year)>0:
        #             for country in intersection_imporatnt_countries_experences_poor_year:
        #                 model_table.loc[year, country] = 1
        #             model_table.loc[year, 'number of important countries'] = len(intersection_imporatnt_countries_experences_poor_year)
        #         else:
        #             model_table.loc[year, country] = 0
        #         # important_countries_experences_poor_year.loc[year, 'number of important countries'] = df[(df['Area'] == country)&(df['Year']==year)].percent_yield_anomaly.values[0]
        #     model_table.fillna(0,inplace=True)
        #     model_table.astype(int)
        #     # important_countries_experences_poor_year.style.background_gradient(cmap='gray_r')
        #     # Calculate the pool year frequency for each country
        #
        #     model_table.loc['Row_sum'] = model_table.apply(lambda x:x.sum())
        #     important_countries_experences_poor_year.loc['Row_sum']=model_table.loc['Row_sum']
        #
        #     # dstyle=important_countries_experences_poor_year.style.applymap(highlight_5)
        #      # # dstyle=dstyle.apply(highlight_0,subset=['number of important countries']).apply(highlight_0,subset=pd.IndexSlice[['Row_sum'],:])
        #
        #     cm = sns.light_palette("green", as_cmap=True)
        #     dstyle = important_countries_experences_poor_year.style.background_gradient(cmap=cm, axis=1, subset=['number of important countries'])
        #     # Use mask to set style
        #     dstyle = dstyle.apply(lambda _: model_table.applymap(color_boolean),axis=None)
        #
        #     # df_style = important_countries_experences_poor_year.style.background_gradient(cmap='gray_r')
        #
        #     if ds.equals(dslist[0]):
        #         dfi.export(obj=dstyle,
        #                    filename=r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly -5%' + '.png', fontsize=30)
        #         important_countries_experences_poor_year.to_csv(
        #             r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly -5%' + '.csv')
        #
        #     elif ds.equals(dslist[1]):
        #         dfi.export(obj=dstyle,
        #                    filename=r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly -10%' + '.png',
        #                    fontsize=30)
        #         important_countries_experences_poor_year.to_csv(
        #             r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly -10%' + '.csv')
        #
        #     elif ds.equals(dslist[2]):
        #         dfi.export(obj=dstyle,
        #                    filename=r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly botttom quartile' + '.png',
        #                    fontsize=30)
        #         important_countries_experences_poor_year.to_csv(
        #             r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly bottom quartile' + '.csv')


def calculate_exportation():
    for crop in crops:
        important_countries_list = []
        clos1 = ['crop_type', 'country', 'average_production', 'percent']
        percent_of_important_countries = pd.DataFrame(columns=clos1)
        df = pd.read_csv('data/crop/FAO_exportation.csv')
        # Make sure not to double count China
        df = df[~(df.Area == 'China, Taiwan Province of')]
        df = df[~(df.Area == 'China')]
        df = df[~(df.Area == 'China, Hong Kong SAR')]
        df = df[~(df.Area == 'China, Macao SAR')]

        df['Value'].fillna(0, inplace=True)
        if crop == 'maize':
            df['Item'].replace('Maize (corn)', 'maize', inplace=True)
        if crop == 'wheat':
            df['Item'].replace('Wheat', 'wheat', inplace=True)
        if crop == 'soy':
            df['Item'].replace('Soya beans', 'soy', inplace=True)

        df = df[df['Item'] == crop]
        # identify the important countries
        ds = df.loc[(df['Year'] >= 2015) & (df['Year'] <= 2020)]
        ds.reset_index()
        for country in ds['Area'].drop_duplicates():
            # calculate average total national production over the period 2015-2020
            average_total_production = ds[ds['Area'] == country].Value.sum() / len(ds[ds['Area'] == country])
            percent_of_important_countries.loc[len(percent_of_important_countries)] = [crop, country,
                                                                                       average_total_production,
                                                                                       0]  # ,average_total_production/total_global_production]
        # calculate average total global production
        average_total_global_production = percent_of_important_countries.average_production.sum()

        for country in percent_of_important_countries.country:
            #  calculate the percent of total global crop production produced by each country
            percent_of_important_countries.loc[(percent_of_important_countries.country == country), 'percent'] = float(
                percent_of_important_countries[
                    percent_of_important_countries.country == country].average_production / average_total_global_production)
        # sorted by percent
        percent_of_important_countries.sort_values(by='percent', ascending=False, inplace=True)
        percent_of_important_countries.reset_index()

        # Come up with our list of “important countries”
        if True:
            important_countries_list_after1990 = []
            total_percent = 0
            flag = 0.80
            if crop == 'maize':
                flag = 0.75
            elif crop == 'wheat':
                flag = 0.75
            elif crop == 'soy':
                flag = 0.85
            for country in percent_of_important_countries.country:

                total_percent = total_percent + float(
                    percent_of_important_countries[percent_of_important_countries['country'] == country].percent)
                important_countries_list_after1990.append(country)

                if total_percent > flag:
                    break

        percent_of_important_countries['percent'] = percent_of_important_countries['percent'].apply(
            lambda x: format(x, '.4%'))
        # percent_of_important_countries.to_csv('table/sorted by percent/' + crop + ' percent of important countries exportation.csv')

        # calculate the percent of each year
        df['eptGauExp']=''
        # df['crop_exportation_anomaly'] = ''
        df['percent_exportation_anomaly'] = ''
        df['percent_lower_than_-5%'] = ''
        df['percent_lower_than_-10%'] = ''
        df['percent_bottom_quartile'] = ''
        poor_year_list = []
        for country in df['Area'].drop_duplicates():
            df1 = df.loc[df['Area'] == country]
            df1['eptGauExp']=ndimage.filters.gaussian_filter1d(df1.Value.values, 3)
            df1['percent_exportation_anomaly'] = (df1.Value.values - df1.eptGauExp) / df1.eptGauExp
            df1.fillna(0, inplace=True)
             # Identify every year when the percent crop yield anomaly is s in the bottom 1/4 of all yield anomalies
            df1.sort_values(by='percent_exportation_anomaly', inplace=True, ascending=True)
            df1.loc[df1.head(round(0.25 * len(df1))).index, 'percent_bottom_quartile'] = 'poor'
            # Identify every year when the percent crop yield anomaly is lower than -5% for each country
            df1.loc[(df1.percent_exportation_anomaly < -0.05), 'percent_lower_than_-5%'] = 'poor'
            df1.loc[(df1.percent_exportation_anomaly < -0.10), 'percent_lower_than_-10%'] = 'poor'
            # rewrite df with df1
            df[df['Area'] == country] = df1[df1['Area'] == country]
        # print(important_countries_list)
        df.percent_exportation_anomaly.fillna(0, inplace=True)

        # Identify when there are years with at least 4 important countries that have poor exportation
        ds_5 = df[df['percent_lower_than_-5%'] == 'poor']
        ds_10 = df[df['percent_lower_than_-10%'] == 'poor']
        ds_bottom_quartile = df[df['percent_bottom_quartile'] == 'poor']
        dslist = [ds_5, ds_10, ds_bottom_quartile]
        # dsnamelist=['-5%','-10%','bottom_quartile']

        clos0 = ['crop', 'poor_year', 'number', 'poor_year_important_countries_blew_-5%']
        poor_year_important_countries = pd.DataFrame(columns=clos0)
        clos2 = ['crop', 'year', 'number_of_important_countries_blew_-5%', 'number_of_important_countries_blew_-10%',
                 'number_of_important_countries_bottom_quartile','Gau']
        every_year_important_countries = pd.DataFrame(columns=clos2)
        important_countries_list = Series(important_countries_list_after1990)
        # Time series of the number of important countries that meet our “poor year” threshold for each year
        for ds in dslist:
            ds.sort_values(by='Year', inplace=True)
            for year in ds.Year.drop_duplicates():
                intersection_important_countries = pd.Series(
                    list(set(ds[ds['Year'] == year].Area).intersection(important_countries_list)))
                if ds.equals(dslist[0]):
                    every_year_important_countries.loc[len(every_year_important_countries)] = [crop, year,
                                                                                               len(intersection_important_countries),
                                                                                               0, 0,0]
                elif ds.equals(dslist[1]):
                    every_year_important_countries.loc[every_year_important_countries[
                                                           'year'] == year, 'number_of_important_countries_blew_-10%'] = len(
                        intersection_important_countries)
                else:
                    every_year_important_countries.loc[every_year_important_countries[
                                                           'year'] == year, 'number_of_important_countries_bottom_quartile'] = len(
                        intersection_important_countries)
        every_year_important_countries.fillna(0,inplace=True)
        # every_year_important_countries.to_csv('table/number of countries/'+crop+str(flag)+'Gau number of important countries exportation.csv')
        pore='exportation'
        calculate_the_linear_trends(crop, every_year_important_countries,pore)
    print('123')



def color_boolean(val):
    return f'background-color: {"red" if val else "white"}'


def highlight(val, big_gain, big_loss):
    if val > big_gain:
        return 'background-color: yellow'
    elif val < big_loss:
        return 'background-color: light green'
    else:
        return ''


def running_mean(data, w):
    res = []
    for i in range(len(data) - w + 1):
        ave = 0
        for j in range(w):
            ave += data[i + j]
        ave /= w
        res.append(ave)
    return res


def combine_image(crop):
    # img1 = Image.open(
    #     r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly -5%' + '.png')
    # img2 = Image.open(
    #     r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly -10%' + '.png')
    # img3 = Image.open(
    #     r'C:/Users/mari/Desktop/results/new analysis/test/stacked bar chart/' + crop + 'yield anomaly botttom quartile' + '.png')
    img1 = Image.open(
        'figures/linear trend/' + crop + '75yldAnomGau.png')
    img2 = Image.open(
        'figures/linear trend/'+ crop + '75yldAnomGauremoveIndia'+'.png')
    img3 = Image.open(
        'figures/linear trend/'+ crop + '75yldAnomGauremoveChinaIndia'+'.png')
    img4 = Image.open(
        'figures/linear trend/' + crop + '75yldAnomGauremoveChina' + '.png')
    img5 = Image.open(
        'figures/linear trend/' + crop + '75yldAnomGauremoveAmerica'+'.png')
    img6 = Image.open(
        'figures/linear trend/' + crop + '75yldAnomGauremoveACI.png')

    ims1 = [img1, img2, img3]
    ims2 = [img3, img5, img6]
    width, height = img1.size
    result1 = Image.new(img1.mode, (width, height * 3))
    result2 = Image.new(img4.mode, (width, height * 3))
    for i, im in enumerate(ims1):
        result1.paste(im, box=(0, i * height))
    for i, im in enumerate(ims2):
        result2.paste(im, box=(0, i * height))
    result1.save(
        'figures/linear trend/' + crop + 'yield anomaly combined left' + '.png')
    result2.save(
        'figures/linear trend/' + crop + 'yield anomaly combined right' + '.png')
    png1 = 'figures/linear trend/' + crop + 'yield anomaly combined left' + '.png'
    png2 = 'figures/linear trend/' + crop + 'yield anomaly combined right' + '.png'
    result=join(png1,png2,'horizontal')
    result.save('figures/linear trend/' + crop + 'yield anomaly combined' + '.png')


def calculate_the_poor_years(crop, every_year_important_countries):
    smooth = every_year_important_countries.columns[-1]
    # if smooth == 'yldAnomSmooth5':
    #     return 0
    # if smooth == 'yldAnomSmooth9':
    #     return 0
    list5 = every_year_important_countries[['year', 'number_of_important_countries_blew_-5%']]
    list5.sort_values(by='number_of_important_countries_blew_-5%', inplace=True, ascending=False)
    list5.reset_index(inplace=True)
    list10 = every_year_important_countries[['year', 'number_of_important_countries_blew_-10%']]
    list10.sort_values(by='number_of_important_countries_blew_-10%', inplace=True, ascending=False)
    list10.reset_index(inplace=True)
    listbottom = every_year_important_countries[['year', 'number_of_important_countries_bottom_quartile']]
    listbottom.sort_values(by='number_of_important_countries_bottom_quartile', inplace=True, ascending=False)
    listbottom.reset_index(inplace=True)


    list5_number = list5['number_of_important_countries_blew_-5%'].loc[0:5]
    list5Min = min(list5_number)
    list5 = list5[list5['number_of_important_countries_blew_-5%'] >= list5Min]
    year5 = list5['year'].to_list()
    list10_number = list10['number_of_important_countries_blew_-10%'].loc[0:2]
    list10Min = min(list10_number)
    list10 = list10[list10['number_of_important_countries_blew_-10%'] >= list10Min]
    year10 = list10['year'].to_list()

    listbottom_number = listbottom['number_of_important_countries_bottom_quartile'].loc[0:9]
    listbottomMin = min(listbottom_number)
    listbottom = listbottom[listbottom['number_of_important_countries_bottom_quartile'] >= listbottomMin]
    yearbottom = listbottom['year'].to_list()

    yearlist = list(set(year5).intersection(yearbottom))
    yearlist = list(set(yearlist).intersection(year10))
    if len(yearlist)>=6:
        listbottom = listbottom[listbottom['number_of_important_countries_bottom_quartile'] >listbottomMin]
        yearbottom = listbottom['year'].to_list()
        yearlist = list(set(year5).intersection(yearbottom))
        yearlist = list(set(yearlist).intersection(year10))
        if len(yearlist)>=6:
            list10 = list10[~(list10['number_of_important_countries_blew_-10%'] == 0)]
            year10 = list10['year'].to_list()
            yearlist = list(set(year5).intersection(yearbottom))
            yearlist = list(set(yearlist).intersection(year10))
            if len(yearlist)>=6:
                sumlist=[]
                yearlisto=[]

                for year in yearlist:
                    sum = every_year_important_countries[every_year_important_countries['year']==year]['number_of_important_countries_blew_-10%'].values+every_year_important_countries[every_year_important_countries['year']==year]['number_of_important_countries_blew_-5%'].values+every_year_important_countries[every_year_important_countries['year']==year].number_of_important_countries_bottom_quartile.values
                    sumlist.append(sum)

                min_sum=sumlist[0]
                for sum in sumlist:
                    if sum<= min_sum:
                        min_sum=sum

                for sum in sumlist:
                    if sum==min_sum:
                        yearlist.pop(sumlist.index(sum))
                # yearlist=yearlisto



    return yearlist


def calculate_the_linear_trends(crop, every_year_important_countries,pore):
    fig = plt.figure()
    X = every_year_important_countries[['year']]
    Y1 = every_year_important_countries['number_of_important_countries_blew_-5%']
    Y2 = every_year_important_countries['number_of_important_countries_blew_-10%']
    Y3 = every_year_important_countries['number_of_important_countries_bottom_quartile']
    smooth = every_year_important_countries.columns[-1]
    # plt.scatter(X,Y1)

    regr = LinearRegression()
    regr1 = LinearRegression()
    regr2 = LinearRegression()
    regr.fit(X, Y1)
    regr1.fit(X, Y2)
    regr2.fit(X, Y3)
    plt.plot(X, regr.predict(X), color='red', label='blew_-5%')
    plt.plot(X, regr1.predict(X), color='blue', label='blew_-10%')
    plt.plot(X, regr2.predict(X), color='green', label='bottom_quartile')
    plt.plot(every_year_important_countries.year,
             every_year_important_countries['number_of_important_countries_blew_-5%'],
             linewidth=1, color='red')
    plt.plot(every_year_important_countries.year,
             every_year_important_countries['number_of_important_countries_blew_-10%'],
             linewidth=1, color='blue')
    plt.plot(every_year_important_countries.year,
             every_year_important_countries.number_of_important_countries_bottom_quartile,
             linewidth=1, color='green')
    plt.xlabel('year')
    plt.ylabel('number')
    plt.legend()
    plt.title(crop + ' ' + pore+'anomaly')
    percent = 80
    if crop == 'maize':
        percent = 75
    if crop == 'wheat':
        percent = 75
    if crop=='soy':
        percent=0.85
    fig.savefig('figures/linear trend/' + crop + str(percent) + smooth + pore+'.png')
    # plt.show()


def time_series_of_yield_anomalies(crop, define_method, important_countries, df, every_year_important_countries,
                                   model_table):
    fig = plt.figure(figsize=(15.25, 6))
    ax = plt.subplot(111)
    yearRange = [1961, 2020]
    years = np.arange(yearRange[0], yearRange[1]+1, dtype='int')
    # years = df['year'].drop_duplicates()
    smooth = every_year_important_countries.columns[-1]
    x = years  # .to_list() #1-D array of years
    # x = x.astype(np.float64).to_numpy()
    y = important_countries  # 1-D array from 1 to # of regions
    y = y.to_numpy()
    regions = important_countries
    yldAnomArr0 = pd.DataFrame(index=y, columns=x)
    regions = important_countries

    for country in regions:
        for year in years:
            yldAnomArr0.loc[country, year] = df[(df['year'] == year) & (df['country'] == country)][smooth].values[0]
    # convert dataframe to array
    yldAnomArr0 = np.array(yldAnomArr0.values)
    # create a empty array object
    yldAnomArr = np.zeros([y.size, x.size]) * np.nan
    #  set values from datframe to array
    for i, value in np.ndenumerate(yldAnomArr0):
        # for yrInd in yldAnomArr.columns:
        yldAnomArr[i] = value * 100
    # The dimensions of yldAnomArr should correspond to the sizes of x and y

    # plot the anomalies as a matrix
    # x, y = np.meshgrid(years, range(regions.size))
    ax.pcolor(x, y, yldAnomArr, cmap='BrBG', vmin=-20, vmax=20)
    # plt.show()
    # if there are missing values mark those with hatching. If no values are missing, you can skip this
    # zm = np.ma.masked_where(np.isfinite(yldAnomArr), yldAnomArr)
    # zm = np.zeros([x.size, y.size]) * 0
    # # zm[np.isnan(zm)] = 1
    model_table_index = model_table.index.to_list()
    model_table_columns = model_table.columns.to_list()
    # model_table_T=pd.DataFrame(model_table.values.T,index=model_table_index,columns=model_table_columns)
    # model_table = np.array(model_table)
    indexlist = []

    for year in model_table_index:
        for country in model_table_columns:
            if model_table.loc[year, country] == 1:
                # zm[i]=1
                regionIndex = [i for i, j in enumerate(model_table_columns) if j == country]
                ic = [year, regionIndex[0]]
                indexlist.append(ic)
    iR = []
    yrInd = []
    for arr in indexlist:
        yrInd.append(arr[0])
        iR.append(arr[1])
    iR = np.array(iR)
    yrInd = np.array(yrInd)
    ax.scatter(yrInd, iR, color='red', s=100, alpha=0.5)
    norm4 = Normalize(vmin=-20, vmax=20, clip=False)
    pct_map = cm.ScalarMappable(norm=norm4, cmap='BrBG')  # define the colormap
    plt.colorbar(pct_map, label=smooth, extend='both')

    # ax.set_xticks(years[9::10]) #place an x-tick every 10 years
    # ax.set_ylim(0, regions.size) #set y-limit
    # ax.set_yticks(np.array(range(regions.size))) #set y-ticks
    # ax.set_yticklabels(regions) #set y-labels
    plt.title(crop + smooth)
    fig.savefig('figures/time series with red dot/' + crop + ' ' + smooth + ' ' + define_method+ '.png')


def crop_yield_anomalies_spatial_map(crop, df, pooryears):

    smooth = df.columns[-4]
    ax0 = plt.subplot(111, projection=ccrs.PlateCarree())
    # next define colorbar normalization as between -.2 or .2 (e.g. for fractional yield anomalies between -20% and 20% of normal)
    norm4 = Normalize(vmin=-.2, vmax=.2, clip=False)
    pct_map = cm.ScalarMappable(norm=norm4, cmap='BrBG')  # define the colormap
    # define the figure object
    fig = plt.figure(0)
    # Plot the probability differences as a map
    ccrs.PlateCarree()
    ax0 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree());
    ax0.coastlines(resolution='50m', zorder=2.5);
    reader = shpreader.Reader('data/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
    worldShps = list(reader.geometries())
    ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())

    ax0.add_feature(ADM0, facecolor='none', edgecolor='k', linewidth=0.5, zorder=2.3)
    ax0.add_feature(cartopy.feature.BORDERS, zorder=2.5)
    # ax0.set_extent(largeReg, crs=ccrs.PlateCarree(central_longitude=0.0))
    ax0.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k', facecolor='powderblue')
    plt.colorbar(pct_map, label=smooth)
    # Read in the location shape data that will match the
    with open("data/plotting_objects/fao_locs_shps_msks.pickle", "rb") as input_file:
        locDict = cPickle.load(input_file)
        # get a list of country names from your FAO shapefile
        # names = list of country names in the year you want to plot
    for year in pooryears:
        names = df[df['year'] == year].country.drop_duplicates()
        countries_keys = list(locDict.keys())
        # Next loop through the names list to find them in your dataframe you are plotting (e.g. FAO percent yield anomalies for a given year)
        for ix, ixname in enumerate(names):

            # test whether “ixname” is in your dataframe of values to plot, and continue if not
            # something like “ if np.size( dataframe size when selecting on ixname )==0:continue
            if ixname not in countries_keys:
                continue
            # here we can save time because I’ve already made a dictionary object to link the country names to the needed shapefiles
            segs = locDict[ixname][1]
            print(ixname)
            plotVal = df[(df['year'] == year) & (df['country'] == ixname)][smooth].values[
                0]  # Here you need to locate the value to plot based on your dataframe (e.g. FAO percent yield anomalies for a given year)

            # The “segs” refer to the raster objects that make up the country. Sometimes it will only be 1, and sometimes it will be multiple, so we need to be able to deal with that
            for ijx in range(np.size(segs)):
                if np.size(segs) > 1:
                    adm = segs[ijx]
                else:
                    adm = segs[0]
                # call up the figure object to plot on it
                plt.figure(0)
                plt.gca()
                # this is finally the programming that actually plots the value on the map
                ax0.add_feature(adm, facecolor=pct_map.to_rgba(plotVal),
                                edgecolor='k')  # color the segment according to the value
        # plt.show()
        plt.title(crop + '_' + str(year) + smooth)

        fig.savefig('figures/spatial map/' + crop + str(year) + smooth + '.png')


def time_series_of_poor_year_over_time(crop, every_year_important_countries):
    fig = plt.figure()
    smooth = every_year_important_countries.columns[-1]
    plt.subplot(211)
    plt.plot(every_year_important_countries.year,
             every_year_important_countries['number_of_important_countries_blew_-5%'], label='blew_-5%',
             linewidth=1, color=np.random.rand(3, ))
    plt.plot(every_year_important_countries.year,
             every_year_important_countries['number_of_important_countries_blew_-10%'], label='blew_-10%',
             linewidth=1, color=np.random.rand(3, ))
    plt.plot(every_year_important_countries.year,
             every_year_important_countries.number_of_important_countries_bottom_quartile, label='bottom_quartile',
             linewidth=1, color=np.random.rand(3, ))
    plt.legend()
    plt.grid(True, axis='y')

    plt.legend()
    plt.title(crop + ' ' + 'Yield anomaly')
    plt.ylabel('number of important countries')
    plt.xlabel('Year')
    plt.tight_layout()
    fig.savefig('data/linear trend/' + crop + smooth + ' ' + 'yield anomaly ' + 'number of important countries every year' + ' ' + '.png')


def replace_yield_anomalies_with0(df):
    smooth = df.columns[-1]
    if smooth=='yldAnomSmooth5':
        df['yldAnomSmooth5'][df['yldAnomSmooth5']>0]=0
    if smooth == 'yldAnomSmooth9':
        df['yldAnomSmooth9'][df['yldAnomSmooth9'] > 0] = 0
    if smooth == 'yldAnomGau':
        df['yldAnomGau'][df['yldAnomGau'] > 0] = 0
    return df


def join(png1, png2, flag='horizontal'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:  a combined picture
    """
    img1, img2 = Image.open(png1), Image.open(png2)
    # Uniform picture size, you can customize settings (width, height)
    # img1 = img1.resize((1500, 1000), Image.ANTIALIAS)
    # img2 = img2.resize((1500, 1000), Image.ANTIALIAS)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save('horizontal.png')
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save('vertical.png')
    return joint


def calculate_censored_percent_yield_anomaly_time_series (crop,important_countries,df ):
    fig = plt.figure()
    # set all crop yield anomalies that are greater than 0 equal to 0
    # df = replace_yield_anomalies_with0(df)
    important_countries = important_countries.to_list()
    X = df[['year']].drop_duplicates()
    smooth = df.columns[-4]
    if smooth=='yldAnomGau':
        df['yldAnomGau'][df['yldAnomGau'] > 0] = 0
        for country in important_countries:
            Y = df[df['country']==country]['yldAnomGau']
            regr = LinearRegression()
            regr.fit(X, Y)
            plt.plot(X, regr.predict(X), color=np.random.rand(3, ), linewidth=1,label=country)

        plt.xlabel('year')
        plt.ylabel('number')
        plt.legend()
        plt.title(crop + ' ' + 'Yield anomaly every country')
        percent = 80
        if crop == 'maize':
            percent = 75
        if crop == 'wheat':
            percent = 75
        if crop == 'soy':
            percent = 0.85
        fig.savefig('data/linear trend/' + crop + str(percent) + smooth + 'censoredyieldanomalies.png')


calculate_exportation()
# calculate_production()
