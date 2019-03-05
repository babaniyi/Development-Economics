# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:25:16 2019

@author: User
"""
##### Panel data
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
pd.options.display.float_format = '{:,.2f}'.format

#=========================================================================================================
#%% Question 1
data = pd.read_stata('C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Development/Solutions/Thesis-Heterogeneities-Risk-Insurance-Uganda-master/panel/dataUGA.dta')
data = data[['age','age_sq','ethnic','female','urban','hh','ctotal','year','familysize','inctotal','wave']]
data['lnc'] = np.log(data['ctotal'])
#data['lnN'] = np.log(data['familysize'])
data['ln_inc'] = np.log(data['inctotal'])
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
totalconsumption = data[['wave', 'ctotal']]
totalconsumption = data.groupby(by='wave')[['ctotal']].sum()
totalconsumption['wave'] = totalconsumption.index
totalconsumption.columns = ['aggC', 'wave']
totalconsumption['aggC']  = np.log(totalconsumption['aggC'])

data = data.merge(totalconsumption, on ='wave', how='left')

import statsmodels.formula.api as smf
olsr1 = smf.ols(formula="ctotal ~ age +age_sq +familysize+C(ethnic)+female+urban", data=data).fit()
print(olsr1.summary())
data["c_res"] = olsr1.resid
data['ln_cres'] = np.log(data['c_res'])
data['ln_cres'] = data['ln_cres'].replace([np.inf, -np.inf], np.nan)
#data.loc[data["ln_cres"]==-np.inf, "ln_cres"] = np.nan #replace inf with nan
data['ln_cres'] = data['ln_cres'].fillna(data['ln_cres'].mean()) #Fill NA's with mean

# Income residuals
olsr2 = smf.ols(formula="inctotal ~ age +age_sq +familysize+ C(ethnic)+female+urban", data=data).fit()
print(olsr2.summary())
data["inc_res"] = olsr2.resid
data['ln_incres'] = np.log(data['inc_res'])
data['ln_incres'] = data['ln_incres'].replace([np.inf, -np.inf], np.nan)
data['ln_incres'] = data['ln_incres'].fillna(data['ln_cres'].mean()) #Fill NA's with mean

dat = data.copy()
# Panel
data.sort_values(["hh","wave"])
data.set_index(["hh","wave"],inplace=True)
datadiff = data.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
datadiff.columns = ["Δc","Δy","Δc_bar"]
datadiff.reset_index(inplace=True)
datadiff = datadiff[datadiff.wave != "2009-2010"]

olsr3 = smf.ols(formula=" Δc~Δy+Δc_bar ", data=datadiff).fit()
print(olsr3.summary())

olsr3.params

#%% QUESTION 2 
#% First clear the data 

avginc = dat[['wave', 'inctotal']]
avginc = avginc.groupby(by='wave')[['inctotal']].sum()
avginc['wave'] = avginc.index
avginc.columns = ['avI', 'wave']
avginc['avI']  = np.log(avginc['avI'])
avginc['avI'].describe()

hhmeans = dat.groupby(by=["wave","hh"])[["ln_inc"]].mean()
hhmeans.reset_index(inplace=True)
hhmeans.rename(columns={"ln_inc":"avgc"}, inplace=True)
hhmeans.loc[hhmeans["avgc"]==-np.inf, "avgc"] = np.nan
hhmeans["avgc"].fillna(np.mean(hhmeans["avgc"]), inplace=True)
dat = dat.merge(hhmeans, on=["wave","hh"], how="outer")

dat['quantile'] = pd.qcut(dat['avgc'], 5, labels=False)
dat['pct_rank'] = dat.avgc.rank(pct=True)

# group income into quantiles
qmeans = dat[["quantile"]] 
qmeans = dat.groupby(by=["quantile"])[["avgc"]].mean()
qmeans.reset_index(inplace=True)
qmedian  = dat.groupby(by="quantile")[["avgc"]].median()
qmedian.reset_index(inplace=True)
qmedian.columns = ["quantile","av_med"]
qgroup = qmeans.merge(qmedian, on=["quantile"], how="outer")
qgroup['quantile'] = qgroup['quantile'].replace([0,1,2,3,4],['bottom20%','20-40%','40-60%','60-80%','top20%'])

import matplotlib.pyplot as plt
plt.plot(qgroup['quantile'],qgroup['avgc'],label='Mean');
plt.plot(qgroup['quantile'],qgroup['av_med'],label='Median');
plt.ylabel('Income')
plt.title('Income groups')
plt.grid(True)
plt.legend()
plt.savefig('incgroup.jpeg')
plt.show()


dat0 = dat[dat["quantile"]==0]
dat1 = dat[dat["quantile"]==1]
dat2 = dat[dat["quantile"]==2]
dat3 = dat[dat["quantile"]==3]
dat4 = dat[dat["quantile"]==4]

# Bottom 20%
dat0.sort_values(["hh","wave"])
dat0.set_index(["hh","wave"],inplace=True)
dat0 = dat0.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
dat0.columns = ["dc","dy","dcb"]
dat0.reset_index(inplace=True)
dat0 = dat0[dat0.wave != "2009-2010"]
dat0.to_csv("dat0.csv", index=False)
olsr4 = smf.ols(formula=" dc~dy+dcb ", data=dat0).fit()
print(olsr4.summary())

# Bottom 20 - 40%
dat1.sort_values(["hh","wave"])
dat1.set_index(["hh","wave"],inplace=True)
dat1 = dat1.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
dat1.columns = ["dc","dy","dcb"]
dat1.reset_index(inplace=True)
dat1 = dat1[dat1.wave != "2009-2010"]
dat1.to_csv("dat1.csv", index=False)
olsr5 = smf.ols(formula=" dc~dy+dcb ", data=dat1).fit()
print(olsr5.summary())

# 40-60%
dat2.sort_values(["hh","wave"])
dat2.set_index(["hh","wave"],inplace=True)
dat2 = dat2.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
dat2.columns = ["dc","dy","dcb"]
dat2.reset_index(inplace=True)
dat2 = dat2[dat2.wave != "2009-2010"]
dat2.to_csv("dat2.csv", index=False)
olsr6 = smf.ols(formula=" dc~dy+dcb ", data=dat2).fit()
print(olsr6.summary())

#60-80%
dat3.sort_values(["hh","wave"])
dat3.set_index(["hh","wave"],inplace=True)
dat3 = dat3.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
dat3.columns = ["dc","dy","dcb"]
dat3.reset_index(inplace=True)
dat3 = dat3[dat3.wave != "2009-2010"]
dat3.to_csv("dat3.csv", index=False)
olsr7 = smf.ols(formula=" dc~dy+dcb ", data=dat3).fit()
print(olsr7.summary())

# top 20%
dat4.sort_values(["hh","wave"])
dat4.set_index(["hh","wave"],inplace=True)
dat4 = dat4.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
dat4.columns = ["dc","dy","dcb"]
dat4.reset_index(inplace=True)
dat4 = dat4[dat4.wave != "2009-2010"]
dat4.to_csv("dat4.csv", index=False)
olsr8 = smf.ols(formula=" dc~dy+dcb ", data=dat4).fit()
print(olsr8.summary())

#%% Question 2B
fulldata = pd.read_csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Development/Freshair/UGA_2013_UNPS_v01_M_STATA8/UGA_2013_UNPS_v01_M_STATA8/fulldata.csv")
fulldata['logw']  = np.log(fulldata['totW'])
fulldata = fulldata.replace([np.inf, -np.inf], np.nan)
fulldata['logw'] = fulldata['logw'].fillna(fulldata['logw'].mean()) #Fill NA's with mean
fulldata['logw'].describe()

fulldata['logi']  = np.log(fulldata['inctotal'])
fulldata = fulldata.replace([np.inf, -np.inf], np.nan)
fulldata['logi'] = fulldata['logi'].fillna(fulldata['logi'].mean()) #Fill NA's with mean
fulldata['logi'].describe()

fulldata['logc']  = np.log(fulldata['ctotal'])
fulldata = fulldata.replace([np.inf, -np.inf], np.nan)
fulldata['logc'] = fulldata['logc'].fillna(fulldata['logc'].mean()) #Fill NA's with mean
fulldata['logc'].describe()

olsr9 = sm.ols(formula=" logc~logw", data=fulldata).fit()
print(olsr9.summary())

fulldata['quantile'] = pd.qcut(fulldata['logw'], 5, labels=False)
# group income into quantiles
qmeans = fulldata[["quantile"]] 
qmeans = fulldata.groupby(by=["quantile"])[["logw"]].mean()
qmeans.reset_index(inplace=True)
qmedian  = fulldata.groupby(by="quantile")[["logw"]].median()
qmedian.reset_index(inplace=True)
qmedian.columns = ["quantile","av_med"]
qgroup = qmeans.merge(qmedian, on=["quantile"], how="outer")
qgroup['quantile'] = qgroup['quantile'].replace([0,1,2,3,4],['bottom20%','20-40%','40-60%','60-80%','top20%'])

import matplotlib.pyplot as plt
plt.plot(qgroup['quantile'],qgroup['logw'],label='Mean');
plt.plot(qgroup['quantile'],qgroup['av_med'],label='Median');
plt.ylabel('Wealth')
plt.title('Wealth groups')
plt.grid(True)
plt.legend()
plt.savefig('C:/Users/User/Documents/wtgroup.jpeg')
plt.show()

del qgroup
del hhmeans
del qmeans
del qmedian

#%% QUESTION FOUR
# RURAL
dat_r = dat[dat['urban']==0]
dat_u = dat[dat['urban']==1]

dat_r.sort_values(["hh","wave"])
dat_r.set_index(["hh","wave"],inplace=True)
datadiff_r = dat_r.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
datadiff_r.columns = ["Δc","Δy","Δc_bar"]
datadiff_r.reset_index(inplace=True)
datadiff_r = datadiff_r[datadiff_r.wave != "2009-2010"]

olsr10 = smf.ols(formula=" Δc~Δy+Δc_bar ", data=datadiff_r).fit()
print(olsr10.summary())
datadiff_r.to_csv("datadiff_r.csv",index=False)

dat_r = dat[dat['urban']==0]
hhmeans = dat_r.groupby(by=["wave","hh"])[["ln_inc"]].mean()
hhmeans.reset_index(inplace=True)
hhmeans.rename(columns={"ln_inc":"avgc"}, inplace=True)
hhmeans.loc[hhmeans["avgc"]==-np.inf, "avgc"] = np.nan
hhmeans["avgc"].fillna(np.mean(hhmeans["avgc"]), inplace=True)
dat_r = dat_r.merge(hhmeans, on=["wave","hh"], how="outer")
dat_r['quantile'] = pd.qcut(dat_r['avgc'], 5, labels=False)

# group income into quantiles
qmeans = dat_r[["quantile"]] 
qmeans = dat_r.groupby(by=["quantile"])[["avgc"]].mean()
qmeans.reset_index(inplace=True)
qmedian  = dat_r.groupby(by="quantile")[["avgc"]].median()
qmedian.reset_index(inplace=True)
qmedian.columns = ["quantile","av_med"]
qgroup = qmeans.merge(qmedian, on=["quantile"], how="outer")
qgroup['quantile'] = qgroup['quantile'].replace([0,1,2,3,4],['bottom20%','20-40%','40-60%','60-80%','top20%'])

import matplotlib.pyplot as plt
plt.plot(qgroup['quantile'],qgroup['avgc'],label='Mean');
plt.plot(qgroup['quantile'],qgroup['av_med'],label='Median');
plt.ylabel('Income')
plt.title('Income groups')
plt.grid(True)
plt.legend()
#plt.savefig('incgroup.jpeg')
plt.show()

del qgroup, totalconsumption
del qmeans, hhmeans, qmedian

# URBAN
datu = dat_u.copy()
dat_u.sort_values(["hh","wave"])
dat_u.set_index(["hh","wave"],inplace=True)
datadiff_u = dat_u.groupby(level=0)['ln_cres','ln_incres','aggC',].diff()
datadiff_u.columns = ["Δc","Δy","Δc_bar"]
datadiff_u.reset_index(inplace=True)
datadiff_u = datadiff_u[datadiff_u.wave != "2009-2010"]

olsr11 = smf.ols(formula=" Δc~Δy+Δc_bar ", data=datadiff_u).fit()
print(olsr11.summary())
datadiff_u.to_csv("datadiff_u.csv",index=False)

dat_u = datu[datu['urban']==0]
hhmeans = dat_u.groupby(by=["wave","hh"])[["ln_inc"]].mean()
hhmeans.reset_index(inplace=True)
hhmeans.rename(columns={"ln_inc":"avgc"}, inplace=True)
hhmeans.loc[hhmeans["avgc"]==-np.inf, "avgc"] = np.nan
hhmeans["avgc"].fillna(np.mean(hhmeans["avgc"]), inplace=True)
dat_u = dat_u.merge(hhmeans, on=["wave","hh"], how="outer")
dat_u['quantile'] = pd.qcut(dat_u['avgc'], 5, labels=False)

# group income into quantiles
qmeans = dat_u[["quantile"]] 
qmeans = dat_u.groupby(by=["quantile"])[["avgc"]].mean()
qmeans.reset_index(inplace=True)
qmedian  = dat_u.groupby(by="quantile")[["avgc"]].median()
qmedian.reset_index(inplace=True)
qmedian.columns = ["quantile","av_med"]
qgroup = qmeans.merge(qmedian, on=["quantile"], how="outer")
qgroup['quantile'] = qgroup['quantile'].replace([0,1,2,3,4],['bottom20%','20-40%','40-60%','60-80%','top20%'])

import matplotlib.pyplot as plt
plt.plot(qgroup['quantile'],qgroup['avgc'],label='Mean');
plt.plot(qgroup['quantile'],qgroup['av_med'],label='Median');
plt.ylabel('Income')
plt.title('Income groups')
plt.grid(True)
plt.legend()
#plt.savefig('incgroup.jpeg')
plt.show()

del qgroup, totalconsumption
del qmeans, hhmeans, qmedian
