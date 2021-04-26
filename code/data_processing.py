import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import sys

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Read in the Dataset
dailyData = pd.read_csv("C:/Users/15467/Desktop/00_repository/data/train.csv")
dailyData_test = pd.read_csv("C:/Users/15467/Desktop/00_repository/data/test.csv")
# shape of the dataset
print(dailyData.shape)
print(dailyData_test.shape)


# sample of first few rows
# print(dailyData.head(2))

# variables data type
# print(dailyData.dtypes)

# creating new columns from 'datetime' column
dailyData["date"] = dailyData.datetime.apply(lambda x: x.split()[0])
dailyData["hour"] = dailyData.datetime.apply(lambda x: x.split()[1].split(":")[0])
dailyData["weekday"] = dailyData.date.apply(
    lambda dateString: calendar.day_name[datetime.strptime(dateString, "%Y-%m-%d").weekday()])
dailyData["month"] = dailyData.date.apply(
    lambda dateString: calendar.month_name[datetime.strptime(dateString, "%Y-%m-%d").month])
dailyData["season"] = dailyData.season.map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})
dailyData["weather"] = dailyData.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",
                                              2: " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ",
                                              3: " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
                                              4: " Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog "})

# print(dailyData.head(2))
# print(dailyData.dtypes)
"""[2 rows x 16 columns]
datetime       object   season         object   holiday         int64   workingday      int64
weather        object   temp          float64   atemp         float64   humidity        int64
windspeed     float64   casual          int64   registered      int64   count           int64
date           object   hour           object   weekday        object   month          object"""

# corecing to category type
categoryVariableList = ["hour", "weekday", "month", "season", "weather", "holiday", "workingday"]
for var in categoryVariableList:
    dailyData[var] = dailyData[var].astype("category")

# Dropping Unncessary Columns
dailyData = dailyData.drop(["datetime"], axis=1)
# print(dailyData.dtypes)

# Start With Very Simple Visualization Of Variables DataType Count
# print("类别计数统计：")
# print(dailyData.dtypes.value_counts())  # 类别计数
# print("-------------")

dataTypeDf = pd.DataFrame(dailyData.dtypes.value_counts()).reset_index().rename(
    columns={"index": "variableType", 0: "count"})

print(dataTypeDf)

# 显示类别数柱状图，以下程序跑不通
# begin1-----------------------------
# fig, ax = plt.subplots()
# fig.set_size_inches(12, 6)
# sn.barplot(data=dataTypeDf, x="variableType", y="count", ax=ax)
# ax.set(xlabel='variableTypeariable Type', ylabel='Count', title="Variables DataType Count")
# end1-----------------------------

# missing values analysis
# 本数据集中不存在缺失值

# Skewness In Distribution 分布偏度
msno.matrix(dailyData, figsize=(12, 8))

# Outliers Analysis 离群值分析
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 8)
sn.boxplot(data=dailyData, y="count", orient="v", ax=axes[0][0])
sn.boxplot(data=dailyData, y="count", x="season", orient="v", ax=axes[0][1])
sn.boxplot(data=dailyData, y="count", x="hour", orient="v", ax=axes[1][0])
sn.boxplot(data=dailyData, y="count", x="workingday", orient="v", ax=axes[1][1])

axes[0][0].set(ylabel='Count', title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count', title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count', title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count', title="Box Plot On Count Across Working Day")

# Remove Outliers In The Count Column
dailyDataWithoutOutliers = dailyData[
    np.abs(dailyData["count"] - dailyData["count"].mean()) <= (3 * dailyData["count"].std())]
print("Shape Of The Before Ouliers: ", dailyData.shape)
print("Shape Of The After Ouliers: ", dailyDataWithoutOutliers.shape)

# Correlation Analysis
corrMatt = dailyData[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12, 5)
sn.regplot(x="temp", y="count", data=dailyData, ax=ax1, scatter_kws={"color": "blue"}, line_kws={"color": "black"})
sn.regplot(x="windspeed", y="count", data=dailyData, ax=ax2, scatter_kws={"color": "green"},
           line_kws={"color": "black"})
sn.regplot(x="humidity", y="count", data=dailyData, ax=ax3, scatter_kws={"color": "red"}, line_kws={"color": "black"})

# Visualizing Distribution Of Data
fig, axes = plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(12, 10)
sn.distplot(dailyData["count"], ax=axes[0][0])
stats.probplot(dailyData["count"], dist='norm', fit=True, plot=axes[0][1])
sn.distplot(np.log(dailyDataWithoutOutliers["count"]), ax=axes[1][0])
stats.probplot(np.log1p(dailyDataWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])

# Visualizing Count Vs (Month,Season,Hour,Weekday,Usertype)
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
fig, (ax1, ax2) = plt.subplots(nrows=2)
fig.set_size_inches(12, 20)

sortOrder = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
hueOrder = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

monthAggregated = pd.DataFrame(dailyData.groupby("month")["count"].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by="count", ascending=False)
sn.barplot(data=monthSorted, x="month", y="count", ax=ax1, order=sortOrder)
ax1.set(xlabel='Month', ylabel='Avearage Count', title="Average Count By Month")

hourAggregated = pd.DataFrame(dailyData.groupby(["hour", "season"], sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"], hue=hourAggregated["season"], data=hourAggregated,
             join=True, ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across Season",
        label='big')

fig, (ax3, ax4) = plt.subplots(nrows=2)
fig.set_size_inches(12, 20)

hourAggregated = pd.DataFrame(dailyData.groupby(["hour", "weekday"], sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"], hue=hourAggregated["weekday"], hue_order=hueOrder,
             data=hourAggregated, join=True, ax=ax3)
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across Weekdays",
        label='big')

hourTransformed = pd.melt(dailyData[["hour", "casual", "registered"]], id_vars=['hour'],
                          value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour", "variable"], sort=True)["value"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"], hue=hourAggregated["variable"],
             hue_order=["casual", "registered"], data=hourAggregated, join=True, ax=ax4)
ax4.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across User Type",
        label='big')

plt.show()
