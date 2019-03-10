import pandas as pd
import numpy as np
dollars = 2586.89

#%%Income 

lab9 = pd.read_stata("GSEC8_1.dta")
lab9 = lab9[["HHID","PID","h8q4","h8q30a","h8q30b", "h8q31a","h8q31b","h8q31c","h8q44","h8q44b","h8q45a","h8q45b","h8q45c"]]     
lab9.columns = ["hh","pid","work", "months1","weeks1", "cash1","inkind1", "time1","months2","weeks2", "cash2","inkind2", "time2"]
lab9["pay1"] = lab9.loc[:,["cash1","inkind1"]].sum(axis=1)
lab9["pay2"] = lab9.loc[:,["cash2","inkind2"]].sum(axis=1)
del lab9["cash1"], lab9["inkind1"], lab9["cash2"], lab9["inkind2"]

#Creating week wages
# we take mean average hours and days worked per week: 60 and 6
lab9.loc[lab9.time1 == "Day", 'pay1'] = lab9.loc[lab9.time1 == "Day", 'pay1']*6
lab9.loc[lab9.time1 == "Month", 'pay1'] = lab9.loc[lab9.time1 == "Month", 'pay1']/4
lab9.loc[lab9.time1 == "Hour", 'pay1'] = lab9.loc[lab9.time1 == "Hour", 'pay1']*60

lab9.loc[lab9.time1 == "Day", 'pay2'] = lab9.loc[lab9.time1 == "Day", 'pay2']*6
lab9.loc[lab9.time1 == "Month", 'pay2'] = lab9.loc[lab9.time1 == "Month", 'pay2']/4
lab9.loc[lab9.time1 == "Hour", 'pay2'] = lab9.loc[lab9.time1 == "Hour", 'pay2']*60

#We don't have info about months and weeks worked. We use the sample mean of 2013-2014. Note that this can hidden important inequality on work time.
lab9["wage1"] = lab9.months1*lab9.weeks1*lab9.pay1
lab9["wage2"] = lab9.months2*lab9.weeks2*lab9.pay2


lab99 = lab9.groupby(by="hh")[["wage1","wage2",
                    "months1","weeks1", "time1","months2","weeks2", "time2","pay1","pay2"]].sum()
lab99["wage_total"] = lab99.loc[:,["wage1","wage2,"]].sum(axis=1)
lab99= lab99.replace(0, np.nan)

lab99["hh"] = np.array(lab99.index.values)
summaryw = lab99.describe()/dollars

del lab9

#%% business

bus12 = pd.read_stata('GSEC12.dta')
bus12 = bus12[["hhid","h12q12", "h12q13","h12q15","h12q16","h12q17"]]
bus12.rename(columns={'hhid':'hh'}, inplace=True)
bus12.rename(columns={'h12q13':'revenue'}, inplace=True)

bus12["cost"] = -bus12.loc[:,["h12q15","h12q16","h12q17"]].sum(axis=1)
bus12["bs_profit"] = bus12.loc[:,["revenue","cost"]].sum(axis=1)
bus12["bs_profit"] = bus12["bs_profit"].replace(0,np.nan)
bus12 = bus12[["hh","bs_profit"]]
bus12 = bus12.groupby(by="hh").sum()

bus12["hh"] = np.array(bus12.index.values)

summarybus = bus12.describe()/dollars

#%% Other income

other = pd.read_stata('GSEC11a.dta')
other = other[["HHID","h11q5","h11q6"]]
other.rename(columns={'HHID':'hh'}, inplace=True)
other["other_inc"] = other.loc[:,["h11q5","h11q6"]].sum(axis=1)
other = other[["hh","other_inc"]]
other = other.groupby(by="hh").sum()
other = other
other["hh"] = np.array(other.index.values)
summaryo = other.describe()/dollars


#%% Merge datasets
income_gsec = pd.merge(lab99, bus12, on="hh", how="outer")
income_gsec = pd.merge(income_gsec, other, on="hh", how="outer")
del income_gsec["wage1"], income_gsec["wage2"], bus12,  dollars, other, lab99, summarybus, summaryo, summaryw

sumlab = income_gsec[["wage_total","bs_profit", "other_inc"]].describe()

income_gsec.to_csv('income_hhsec.csv')






