import pandas as pd
import numpy as np
dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF

#%%IMPORT DATA
basic = pd.read_stata('GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","urban","year", "month"]] 
basic.rename(columns={'HHID':'hh'}, inplace=True)
#basic["hh"] = pd.to_numeric(basic["hh"])
pd.value_counts(basic["year"])
pd.value_counts(basic["month"])
basic = basic.dropna(subset = ['year'])
basic["index"] = range(0,len(basic))
basic.set_index(basic["index"], inplace=True)


#%% Consumption
cons = pd.read_csv("cons.csv")
cons = cons[["hh","ctotal","ctotal_dur","ctotal_gift","cfood","cnodur"]]
# ctotal: food + nofood
# ctotal dur: food + nofood + durables
# ctotal gift: food + nofood of gifts
data = pd.merge(basic, cons, on="hh", how="left")

#%% +Wealth
wealth = pd.read_stata('GSEC14A.dta')
wealth = wealth[["HHID","h14q5"]]
wealth.columns = ["HHID","totW"]
wealth.rename(columns={'HHID':'hh'}, inplace=True)
wealth = wealth.groupby(by="hh").sum()
data = pd.merge(data, wealth, on='hh', how='inner')

#%% Income: 
#labor & business income: in US dollars
lab_inc = pd.read_csv('income_hhsec.csv', header=0, na_values='nan')
#Agricultural income: in UG Shillings
ag_inc = pd.read_csv('income_agsec.csv', header=0, na_values='nan')
#del ag_inc["Unnamed: 0"]
inc = pd.merge(lab_inc, ag_inc, on="hh", how="outer")
inc = inc.drop(inc.columns[[0,5]], axis=1)

inc["inctotal"] = inc.loc[:,["wage_total","bs_profit","other_inc","profit_agr","profit_ls","total_agrls"]].sum(axis=1)
inc["inctotal"] = inc["inctotal"].replace(0,np.nan)

suminc1 = inc.describe()/dollars
#Create income share
inc["w_share"] = inc[["wage_total"]].divide(inc.inctotal, axis=0)
inc["agr_share"] = inc[["total_agrls"]].divide(inc.inctotal, axis=0)
inc["bus_share"] = inc[["bs_profit"]].divide(inc.inctotal, axis=0)

data = data.merge( inc, on='hh', how='left')
del ag_inc, lab_inc, inc

#%%
#Store data
data_cwi = data[["hh", "ctotal", "inctotal", "totW"]]


# Trimming 0.1% and 0.1% each tail
for serie in ["ctotal", "totW", "inctotal"]:
    data['percentiles'] = pd.qcut(data[serie], [0.001,0.999], labels = False)
    data.dropna(axis=0, subset=['percentiles'], inplace=True)
    data.drop('percentiles', axis=1, inplace=True)

#%% Import sociodemographic charact
socio = pd.read_csv("sociodem.csv")
socio.drop(socio.columns[0], axis=1, inplace= True)
#socio11.drop_duplicates(subset=['hh'], keep=False)
data = pd.merge(data, socio, on="hh", how="left")
#data = data.drop_duplicates(subset=['hh'], keep=False)

#HH size
health = pd.read_stata('GSEC5.dta', convert_categoricals=False)
health = health[["HHID","PID","h5q4","h5q5","h5q8","h5q10","h5q11","h5q12"]]
familysize =  pd.DataFrame(pd.value_counts(health.HHID))
familysize.columns= ["familysize"]
familysize["hh"] = np.array(familysize.index.values)
data = data.merge(familysize, on="hh", how="left")

sum_c = data[["ctotal","ctotal_dur","ctotal_gift","cfood","cnodur"]].describe()
sum_inc = data[["inctotal","wage_total","bs_profit","profit_agr","profit_ls"]].describe()
sum_lab = data[["agr_share","w_share","bus_share"]].describe()
sum_sociodem = data[["age", "illdays", "urban","familysize"]].describe()


data.to_csv("fulldata.csv", index=False)


