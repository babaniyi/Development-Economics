import pandas as pd
import numpy as np
dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF

# =============================================================================
#  FULL ANALYSIS FOR PROBLEM SET 1
# =============================================================================

data = pd.read_csv("fulldata.csv")
data_urban = data.loc[data["urban"]==1]
data_rural = data.loc[data["urban"]==0]

#%% Question One: Report average CIW per household separately for rural and urban areas.
av_u = np.ceil(data_urban[["hh", "ctotal", "inctotal", "totW"]].describe()/dollars)
av_u
av_r = np.ceil(data_rural[["hh", "ctotal", "inctotal", "totW"]].describe()/dollars)
av_r
np.ceil(data[["hh", "ctotal", "inctotal", "totW"]].describe()/dollars)

#%% Question Two
#=======2. CIW inequality: (1) Show histogram for CIW separately for rural and urban areas; (2) Report
#================the variance of logs for CIW separately for rural and urban areas.

#%% Histogram plot for CIW levels for rural and urban areas . Variance of logs of CIW
import matplotlib.pyplot as plt
from numpy import inf
du = data_urban[["age","ctotal", "inctotal", "totW"]]
du = data_urban[["ctotal", "inctotal", "totW"]].divide(dollars)
du = du[["ctotal", "inctotal", "totW"]].apply(np.log)
du[du == -inf] = 0
du = du.replace(np.nan,0)
du.var()

import seaborn as sns, numpy as np
sns.distplot(du["ctotal"],hist=False , label='Consumption', kde=True)
sns.distplot(du["inctotal"],hist=False , label='Income', kde=True)
sns.distplot(du["totW"],hist=False , label='Welath', kde=True)
plt.legend(loc='upper right')
plt.ylabel('Probability')
plt.title('CIW of Household in Urban Areas')
plt.savefig('urban.png')
plt.show()

d = data_rural[["ctotal", "inctotal", "totW"]].divide(dollars)
d = d[["ctotal", "inctotal", "totW"]].apply(np.log)
d[d == -inf] = 0
d = d.replace(np.nan,0)
d.var()
import seaborn as sns, numpy as np
sns.distplot(d["ctotal"],hist=False , label='Consumption', kde=True)
sns.distplot(d["inctotal"],hist=False , label='Income', kde=True)
sns.distplot(d["totW"],hist=False , label='Welath', kde=True)
plt.legend(loc='upper right')
plt.ylabel('Probability')
plt.title('CIW of Household in Rural Areas')
plt.savefig('rural.png')
plt.show()

df = data[["ctotal", "inctotal", "totW"]].divide(dollars)
df = df[["ctotal", "inctotal", "totW"]].apply(np.log)
df[df == -inf] = 0
df = df.replace(np.nan,0)
df.var()
import seaborn as sns, numpy as np
sns.distplot(df["ctotal"],hist=False , label='Consumption', kde=True)
sns.distplot(df["inctotal"],hist=False , label='Income', kde=True)
sns.distplot(df["totW"],hist=False , label='Welath', kde=True)
plt.legend(loc='upper right')
plt.ylabel('Probability')
plt.title('CIW of all Household in Uganda')
plt.savefig('full1.png')
plt.show()

####===========================================================================#############
#############   Histogram plot of Urban and Rural HH CIW levels ===========================##
############################################################################################
#**** Consumption***************
import seaborn as sns, numpy as np
sns.distplot(du["ctotal"],hist=True , label='Urban', kde=True)
sns.distplot(d["ctotal"],hist=True , label='Rural', kde=True)
plt.legend(loc='upper right')
plt.ylabel('Probability')
plt.xlabel('log Consumption')
plt.title('Consumption')
plt.savefig('csp.png')
plt.show()

#**** Income***************
import seaborn as sns, numpy as np
sns.distplot(du["inctotal"],hist=True , label='Urban', kde=True)
sns.distplot(d["inctotal"],hist=True , label='Rural', kde=True)
plt.legend(loc='upper right')
plt.ylabel('Probability')
plt.xlabel('log Income')
plt.title('Income')
plt.savefig('income.png')
plt.show()

#**** Wealth***************
import seaborn as sns, numpy as np
sns.distplot(du["totW"],hist=True , label='Urban', kde=True)
sns.distplot(d["totW"],hist=True , label='Rural', kde=True)
plt.legend(loc='upper right')
plt.ylabel('Probability')
plt.xlabel('log Wealth')
plt.title('Wealth')
plt.savefig('wealth.png')
plt.show()


#%% Question Three Joint Cross-sectional behavior
#%% Correlation matrix
du.corr(method='pearson') #urban
d.corr(method='pearson') #rural
df.corr(method='pearson') #full

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(du["ctotal"], du["inctotal"], du["totW"], cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('log Income')
ax.set_xlabel('log Consumption')
ax.set_zlabel('log Wealth')
plt.title('Joint Cross-sectional Behavior CIW in Urban')
plt.savefig('teste.png')
plt.show()


#======== Rural========#####
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(d["ctotal"], d["inctotal"], d["totW"], cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('log Income')
ax.set_xlabel('log Consumption')
ax.set_zlabel('log Wealth')
plt.title('Joint Cross-sectional Behavior CIW in Rural')
plt.savefig('teste2.png')
plt.show()

###------------- Full HH --------------#####
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(df["ctotal"], df["inctotal"], df["totW"], cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('log Income')
ax.set_xlabel('log Consumption')
ax.set_zlabel('log Wealth')
plt.title('Joint Cross-sectional Behavior CIW in Uganda')
plt.savefig('teste3.png')
plt.show()


#%% QUESTION FOUR: CIW OVER LIFECYCLE
from numpy import inf
dr = data_rural[["age","ctotal","inctotal","totW"]]
dr = dr.groupby(by="age")[["ctotal","inctotal","totW"]].sum()
dr = dr[["ctotal","inctotal","totW"]].divide(dollars)
dr = dr[["ctotal","inctotal","totW"]].apply(np.log)
dr[dr == -inf] = 0
dr = dr.replace(np.nan, 0)
dr["age"] = np.array(dr.index.values)

d_u = data_urban[["age","ctotal","inctotal","totW"]]
d_u = d_u.groupby(by="age")[["ctotal","inctotal","totW"]].sum()
d_u = d_u[["ctotal","inctotal","totW"]].divide(dollars)
d_u = d_u[["ctotal","inctotal","totW"]].apply(np.log)
d_u[d_u == -inf] = 0
d_u = d_u.replace(np.nan, 0)
d_u["age"] = np.array(d_u.index.values)

d3 = data[["age","ctotal","inctotal","totW"]]
d3 = d3.groupby(by="age")[["ctotal","inctotal","totW"]].sum()
d3 = d3[["ctotal","inctotal","totW"]].divide(dollars)
d3 = d3[["ctotal","inctotal","totW"]].apply(np.log)
d3[d3 == -inf] = 0
d3 = d3.replace(np.nan, 0)
d3["age"] = np.array(d3.index.values)

#**** Wealth***************
from matplotlib import pyplot
pyplot.plot(d_u["age"],d_u["totW"], label='Urban')
pyplot.plot(dr["age"],dr["totW"], label='Rural')
pyplot.plot(d3["age"],d3["totW"], label='Uganda')
pyplot.legend(loc='upper right')
plt.ylabel('log Wealth')
plt.xlabel('Age')
plt.title('Lifecycle Wealth')
plt.grid(True)
pyplot.savefig('wealthhh.png')

from matplotlib import pyplot
pyplot.plot(d_u["age"],d_u["ctotal"], label='Urban')
pyplot.plot(dr["age"],dr["ctotal"], label='Rural')
pyplot.plot(d3["age"],d3["ctotal"], label='Uganda')
pyplot.legend(loc='upper right')
plt.ylabel('log Consumption')
plt.xlabel('Age')
plt.title('Lifecycle Consumption')
plt.grid(True)
pyplot.savefig('clif.png')

from matplotlib import pyplot
pyplot.plot(d_u["age"],d_u["inctotal"], label='Urban')
pyplot.plot(dr["age"],dr["inctotal"], label='Rural')
pyplot.plot(d3["age"],d3["inctotal"], label='Uganda')
pyplot.legend(loc='upper right')
plt.ylabel('log Income')
plt.xlabel('Age')
plt.title('Lifecycle Income')
plt.grid(True)
pyplot.savefig('inlif.png')


#%% QUESTION FIVE
#pd.qcut(d['inctotal'], 5,labels=False)
#[0, .25, .5, .75, 1.]
#pd.qcut(d['inctotal'], [0, .25, .5, .75, 1.],labels=False)

data["w_share"].describe()
data["agr_share"].describe()
data["bus_share"].describe()


pd.qcut(data['inctotal'], 5,labels=False).value_counts(sort=False)
pd.qcut(data['inctotal'], 5,labels=False).value_counts()

b = data[["ctotal", "inctotal", "totW"]]
b = b.divide(dollars)
b = b.apply(np.log)
b = b.groupby(by="inctotal")[["ctotal", "totW"]].sum()
b["inctotal"] = np.array(b.index.values)
b[b == -inf] = 0
b = b.replace(np.nan, 0)

pd.qcut(b.ctotal, 10).value_counts(sort=False)
pd.qcut(b.totW, 5,labels=False).value_counts(sort=False)
#pd.qcut(b.totW, 5).value_counts()


import  numpy  as  np 
import  matplotlib.pyplot  as  plt
#data['inctotal'] = data['inctotal'].replace(0,float('NaN'))
#data = data[["HHID","inctotal","total_W","ctotal"]]
#data = data.dropna()

#===== Plot of the distribution of wealth and consumption of the highest 25% and lowest 25%
N = round(len(b)/4)
incomelow = b.sort_values(['inctotal'], ascending=True).head(N) #With this I take the 25% lowest income.
incomehigh = b.sort_values(['inctotal'], ascending=False).head(N)

plt.plot(incomelow['inctotal'],incomelow['totW'], label = 'Wealth')
plt.plot(incomelow['inctotal'],incomelow['ctotal'], label = 'consumption')
plt.ylabel('Wealth and Consumption')
plt.xlabel('Income')
plt.title("Wealth and Consumption of lowest 25% income")
plt.legend()
plt.savefig('a11')
plt.show()


plt.plot(incomehigh['inctotal'],incomehigh['totW'], label = 'Wealth')
plt.plot(incomehigh['inctotal'],incomehigh['ctotal'], label = 'consumption')
plt.title("Wealth and consumption of highest 25% income")
plt.ylabel('Wealth and Consumption', size = 10)
plt.xlabel('Income', size = 20)
plt.legend()
plt.savefig('a12')
plt.show()







#%% 1.5 CWI shares by percentiles (Inequality a la Piketty)
    
def percentiles_shares(variable1, dataset, percentile = np.array([0,0.01, 0.05, 0.1, 0, 0.2, 0.4, 0.6, 0.8, 1, 0.90, 0.95, 0.99, 1])):
    c_array = np.sort(np.array(dataset[variable1].dropna()))
    c_total = sum(c_array)
    n= len(c_array)
    percentiles = n*percentile
    percentiles = percentiles.tolist()
    percentiles = [int(x) for x in percentiles]
    bottom = percentiles [0:4]
    quintiles = percentiles [4:10]
    top = percentiles [10:15]
    mg_bottom= []
    for i in range(0,len(bottom)):
        a = sum(c_array[bottom[0]:bottom[i]])/c_total
        mg_bottom.append(a)
    mg_quintiles = []
    for i in range(1,len(quintiles)):
        b = sum(c_array[quintiles[i-1]:quintiles[i]])/c_total
        mg_quintiles.append(b)
    mg_top = []
    for i in range(0,len(top)):
        c = sum(c_array[top[i]:top[3]])/c_total
        mg_top.append(c)               
    return mg_bottom, mg_quintiles, mg_top


# percentile shares Uganda------------------------------------
bottom_cwi = []
quintiles_cwi = []
top_cwi = []

for serie in ["ctotal","inctotal","totW"]:
    bottom, quin, top = percentiles_shares(serie, dataset=data)
    bottom_cwi.append(bottom)
    quintiles_cwi.append(quin)
    top_cwi.append(top)

# percentile shares Uganda Rural ----------------------------------
data_rural =  data.loc[data['urban'] == 0] 
bottom_cwi_rur = []
quintiles_cwi_rur = []
top_cwi_rur = []

for serie in ["ctotal","inctotal","totW"]:
    bottom, quin, top = percentiles_shares(serie, dataset= data_rural)
    bottom_cwi_rur.append(bottom)
    quintiles_cwi_rur.append(quin)
    top_cwi_rur.append(top)
  
# percentile shares Uganda Urban ----------------------------------
data_urban =  data.loc[data['urban'] == 1] 
bottom_cwi_urb = []
quintiles_cwi_urb = []
top_cwi_urb = []

for serie in ["ctotal","inctotal","wtotal"]:
    bottom, quin, top = percentiles_shares(serie, dataset= data_urban)
    bottom_cwi_urb.append(bottom)
    quintiles_cwi_urb.append(quin)
    top_cwi_urb.append(top)



#%%SUMMARY STATISTICS
np.ceil(data[["ctotal","ctotal_dur","ctotal_gift","cfood","cnodur"]].describe()/dollars)
np.ceil(data[["inctotal","wage_total","bs_profit","profit_agr","profit_ls"]].describe()/dollars)
np.ceil(data[["age", "illdays", "urban","female","familysize"]].describe()/dollars)










#%% PART TWO
#######=======================================================###########################
###=======  Q1. Redo Question 1 for intensive and extensive margins of labor supply.
######################################################################################
##The employment rate as the fraction of all adults that report being employed or
#have positive hours worked: e = E/N where E is the total number of employed
#adults, and N the total number of adults.
#The hours per worker as the average hours worked in all jobs in the reference week
#among all those who are employed: hw = H/E where H is total hours supplied by
#employed adult workers.

##==== 36 a-g, sum it to get hours worked for all hhh members for job1, 31a and b are cash and gift
# 45b cash of 2nd job, 57-hours, 58 cash

lab = pd.read_stata("GSEC8_1.dta", convert_categoricals=False)
lab = lab[["HHID","PID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g",
            "h8q31a","h8q31b", "h8q45b", "h8q57_2","h8q58a"]] 
    
lab["totalh"] = lab[["h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g","h8q57_2"]].sum(axis=1)
del lab["PID"], lab["h8q36a"],lab["h8q36b"],lab["h8q36c"],lab["h8q36d"],lab["h8q36e"],lab["h8q36f"],lab["h8q36g"],lab["h8q57_2"]
lab["pay"] = lab[["h8q45b","h8q58a","h8q31a","h8q31b"]].sum(axis=1)
del lab["h8q45b"],lab["h8q58a"],lab["h8q31a"],lab["h8q31b"] 
lab.columns = ["hh","totalh","pay"]
lab = lab.groupby(by="hh").sum()
lab = pd.merge(lab,data, on="hh", how="outer")
lab["e"] = lab["hhnum"]/lab["familysize"] ## employment rate
lab["h"] = lab["totalh"]/lab["hhnum"] ## average hours per worker -----intensive
lab["eh"] = lab["e"]*lab["h"] ## hours per adult ----- extensive
lab.to_csv("labordata.csv") ## write to csv

##-------------------------------------------------------------------###########
lab["eh"].describe() # hours per adult in a week
lab["h"].describe() ## average hours per worker
lab["e"].describe() ## employment rate

lab_rural = lab.loc[lab["urban"]==0]
lab_urban = lab.loc[lab["urban"]==1]

lab_rural= lab_rural[["h", "eh"]].apply(np.log)
lab_rural[lab_rural == -inf] = 0

lab_urban= lab_urban[["h", "eh"]].apply(np.log)
lab_urban[lab_urban == -inf] = 0

lab_rural = lab_rural.replace(np.nan,0)
lab_urban = lab_urban.replace(np.nan,0)

##---- labor hours for urban and rural areas
lab_urban["eh"].describe()
lab_urban["h"].describe()

lab_rural["eh"].describe()
lab_rural["h"].describe()

#========= Men and Women --------#############
lab_m = lab.loc[lab["sex"]==1]
lab_f = lab.loc[lab["sex"]==2] # 

lab_m["eh"].describe()
lab_m["h"].describe()

lab_f["eh"].describe()
lab_f["h"].describe()


import seaborn as sns, numpy as np
sns.distplot(lab_m["h"],hist=False , label='Men', kde=True)
sns.distplot(lab_f["h"],hist=False , label='Women', kde=True)
plt.legend(loc='upper right')
plt.xlabel('Hours worked ')
plt.ylabel('Probability')
plt.title('Average hours worked per worker')
plt.savefig('labmen.png')
plt.show()

lab_no = lab.loc[lab["father_educ"]==1] # no formal educ
lab_lp = lab.loc[lab["father_educ"]==2] # less than primary 
lab_p = lab.loc[lab["father_educ"]==3] # completed primary
lab.father_educ = lab.father_educ.replace([5,6,51,61],4)
lab_hs = lab.loc[lab["father_educ"]==4] # completed higher than secondary

lab_no["eh"].describe()
lab_no["h"].describe()

lab_lp["eh"].describe()
lab_lp["h"].describe()

lab_p["eh"].describe()
lab_p["h"].describe()

lab_hs["eh"].describe()
lab_hs["h"].describe()

import seaborn as sns, numpy as np
sns.distplot(lab_no["h"],hist=False , label='No Pry', kde=True)
sns.distplot(lab_lp["h"],hist=True , label='Less than Pry', kde=True)
sns.distplot(lab_p["h"],hist=True , label='Compl. Pry', kde=True)
sns.distplot(lab_hs["h"],hist=True , label='Sec&Higher', kde=True)
plt.legend(loc='upper right')
plt.xlabel('Hours worked ')
plt.ylabel('Probability')
plt.title('Average hours worked per worker')
plt.savefig('labschool.png')
plt.show()



#======== Q2. Redo separately for women and men, and by education groups (less than primary school======
##===      completed, primary school completed, and secondary school completed or higher).
##########################---------------------------------------------#################################
data = pd.read_csv("fulldata.csv")
data_m = data.loc[data["sex"]==1]
data_f = data.loc[data["sex"]==2] # 

#%% Question One: Report average CIW per household separately for MALE and FEMALE.
av_m = data_m[["hh", "ctotal", "inctotal", "totW"]].describe()/dollars
np.ceil(av_m)
av_f = data_f[["hh", "ctotal", "inctotal", "totW"]].describe()/dollars
np.ceil(av_f)

#%% Question Two
import matplotlib.pyplot as plt
from numpy import inf
dm = data_m[["ctotal", "inctotal", "totW"]]
dm = data_m[["ctotal", "inctotal", "totW"]].divide(dollars)
dm = dm[["ctotal", "inctotal", "totW"]].apply(np.log)
dm[dm == -inf] = 0
dm = dm.replace(np.nan,0)
dm.var()

#----- CIW plot for Men
import seaborn as sns, numpy as np
sns.distplot(dm["ctotal"],hist=False , label='Consumption', kde=True)
sns.distplot(dm["inctotal"],hist=False , label='Income', kde=True)
sns.distplot(dm["totW"],hist=False , label='Welath', kde=True)
plt.legend(loc='upper right')
plt.xlabel('log CIW ')
plt.ylabel('Probability')
plt.title('CIW of Men')
plt.savefig('men.png')
plt.show()

df = data_f[["ctotal", "inctotal", "totW"]].divide(dollars)
df = df[["ctotal", "inctotal", "totW"]].apply(np.log)
df[df == -inf] = 0
df = df.replace(np.nan,0)
df.var()
#----- CIW plot for Women
import seaborn as sns, numpy as np
sns.distplot(df["ctotal"],hist=False , label='Consumption', kde=True)
sns.distplot(df["inctotal"],hist=False , label='Income', kde=True)
sns.distplot(df["totW"],hist=False , label='Welath', kde=True)
plt.legend(loc='upper right')
plt.xlabel('log CIW ')
plt.ylabel('Probability')
plt.title('CIW of Women')
plt.savefig('women.png')
plt.show()

#**** Consumption**************
from matplotlib import pyplot
pyplot.hist(dm["ctotal"], label='Men', density=True)
pyplot.hist(df["ctotal"], label='Women', density=True)
pyplot.legend(loc='upper right')
plt.ylabel('Probability')
plt.title('Consumption')
plt.grid(True)
pyplot.savefig('csp_g.png')

#**** Income***************
from matplotlib import pyplot
pyplot.hist(dm["inctotal"], label='Men', density=True)
pyplot.hist(df["inctotal"],label='Women', density=True)
pyplot.legend(loc='upper right')
plt.ylabel('Probability')
plt.title('Income')
plt.grid(True)
pyplot.savefig('inc_g.png')

#**** Wealth***************
from matplotlib import pyplot
pyplot.hist(dm["totW"], label='Men', density=True)
pyplot.hist(df["totW"], label='Women', density=True)
pyplot.legend(loc='upper right')
plt.ylabel('Probability')
plt.title('Wealth')
plt.grid(True)
pyplot.savefig('wealth_g.png')

#%% Correlation matrix
dm.corr(method='pearson') #urban
df.corr(method='pearson') #urban

# library & dataset
import matplotlib.pyplot as plt
import seaborn as sns 

sns.pairplot(dm, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.title('Correlogram of CIW by Men')
plt.savefig('m1.png')
plt.show()

sns.pairplot(df, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.title('Correlogram of CIW by Women')
plt.savefig('m2.png')
plt.show()

#%% QUESTION FOUR: CIW OVER LIFECYCLE

data_m = data.loc[data["sex"]==1]
data_f = data.loc[data["sex"]==2] #


dm = data_m[["age","ctotal","inctotal","totW"]]
dm = dm.groupby(by="age")[["ctotal","inctotal","totW"]].sum()
dm["age"] = np.array(dm.index.values)

df = data_f[["age","ctotal","inctotal","totW"]]
df = df.groupby(by="age")[["ctotal","inctotal","totW"]].sum()
df["age"] = np.array(df.index.values)

import matplotlib.pyplot as plt
from numpy import inf
d_m = data_m[["age","ctotal", "inctotal", "totW"]]
d_m = d_m.groupby(by="age")[["ctotal", "inctotal", "totW"]].sum()
d_m = d_m[["ctotal", "inctotal", "totW"]].divide(dollars)
d_m = d_m[["ctotal", "inctotal", "totW"]].apply(np.log)
d_m[d_m == -inf] = 0

import matplotlib.pyplot as plt
from numpy import inf
d = data_f[["age","ctotal", "inctotal", "totW"]]
d = d.groupby(by="age")[["ctotal", "inctotal", "totW"]].sum()
d = d[["ctotal", "inctotal", "totW"]].divide(dollars)
d = d[["ctotal", "inctotal", "totW"]].apply(np.log)
d[d == -inf] = 0

#**** Wealth***************
from matplotlib import pyplot
pyplot.plot(dm["age"],d_m["totW"], label='Men')
pyplot.plot(df["age"],d["totW"],label='Women')
pyplot.legend(loc='upper right')
plt.ylabel('Wealth')
plt.xlabel('Age')
plt.title('Lifecycle Wealth')
plt.grid(True)
pyplot.savefig('wgen.png')

from matplotlib import pyplot
pyplot.plot(dm["age"],d_m["totW"], label='Men')
pyplot.plot(df["age"],d["totW"], label='Women')
pyplot.legend(loc='upper right')
plt.ylabel('Consumption')
plt.xlabel('Age')
plt.title('Lifecycle Consumption')
plt.grid(True)
pyplot.savefig('clifgen.png')

from matplotlib import pyplot
pyplot.plot(dm["age"],d_m["totW"], label='Men')
pyplot.plot(df["age"],d["totW"],label='Women')
pyplot.legend(loc='upper right')
plt.ylabel('Income')
plt.xlabel('Age')
plt.title('Lifecycle Income')
plt.grid(True)
pyplot.savefig('inlifgen.png')




############################################################################################
#***************** EDUCATION GROUPS *********************************************
#%% PART TWO
data = pd.read_csv("fulldata.csv")
dedu = data
data_no = dedu.loc[dedu["father_educ"]==1] # no formal educ
data_lp = dedu.loc[dedu["father_educ"]==2] # less than primary 
data_p = dedu.loc[dedu["father_educ"]==3] # completed primary
dedu.father_educ = dedu.father_educ.replace([5,6,51,61],4)
data_hs = dedu.loc[dedu["father_educ"]==4] # completed higher than secondary


#%% Question One: Report average CIW per household separately for educational attainment.
np.ceil(data_no[["ctotal", "inctotal", "totW"]].describe()/dollars)
np.ceil(data_lp[["ctotal", "inctotal", "totW"]].describe()/dollars)
np.ceil(data_p[["ctotal", "inctotal", "totW"]].describe()/dollars)
np.ceil(data_hs[["ctotal", "inctotal", "totW"]].describe()/dollars)


#%% Question Two

#%% 
import matplotlib.pyplot as plt
from numpy import inf
dn = data_no[["ctotal", "inctotal", "totW"]].divide(dollars)
dn = dn[["ctotal", "inctotal", "totW"]].apply(np.log)
dn[dn == -inf] = 0
dn.var()

dlp = data_lp[["ctotal", "inctotal", "totW"]].divide(dollars)
dlp = dlp[["ctotal", "inctotal", "totW"]].apply(np.log)
dlp[dlp == -inf] = 0
dlp.var()

dp = data_p[["ctotal", "inctotal", "totW"]].divide(dollars)
dp = dp[["ctotal", "inctotal", "totW"]].apply(np.log)
dp[dp == -inf] = 0
dp.var()

dhs = data_hs[["ctotal", "inctotal", "totW"]].divide(dollars)
dhs = dhs[["ctotal", "inctotal", "totW"]].apply(np.log)
dhs[dhs == -inf] = 0
dhs.var()

#**** Consumption***************
import seaborn as sns, numpy as np
sns.distplot(dn["ctotal"],hist=False , label='No Educ', kde=True)
sns.distplot(dlp["ctotal"],hist=False, label='Less Pry', kde=True)
sns.distplot(dp["ctotal"], hist=False, label='Compl. Pry', kde=True)
sns.distplot(dhs["ctotal"],hist=False, label='Sec&higher', kde=True)
#sns.legend(loc='upper right')
plt.ylabel('Probability')
plt.xlabel('Log Consumption')
plt.title('Consumption by Education group')
plt.grid(True)
pyplot.savefig('csp_ed.png')

#**** Income***************
import seaborn as sns, numpy as np
sns.distplot(dn["inctotal"],hist=False , label='No Educ', kde=True)
sns.distplot(dlp["inctotal"],hist=False, label='Less Pry', kde=True)
sns.distplot(dp["inctotal"], hist=False, label='Compl. Pry', kde=True)
sns.distplot(dhs["inctotal"],hist=False, label='Sec&higher', kde=True)
#sns.legend(loc='upper right')
plt.ylabel('Probability')
plt.xlabel('Log Income')
plt.title('Income by Education group')
plt.grid(True)
pyplot.savefig('inc_ed.png')


#**** Wealth***************
from matplotlib import pyplot
import seaborn as sns, numpy as np
sns.distplot(dn["totW"],hist=False , label='No Educ', kde=True)
sns.distplot(dlp["totW"],hist=False, label='Less Pry', kde=True)
sns.distplot(dp["totW"], hist=False, label='Compl. Pry', kde=True)
sns.distplot(dhs["totW"],hist=False, label='Sec&higher', kde=True)
#sns.legend(loc='upper right')
plt.ylabel('Probability')
plt.xlabel('Log Wealth')
plt.title('Wealth by Education group')
plt.grid(True)
pyplot.savefig('w_ed.png')

#%% Correlation matrix
a = data_no[["ctotal", "inctotal", "totW"]].divide(dollars)
b = data_lp[["ctotal", "inctotal", "totW"]].divide(dollars)
c = data_p[["ctotal", "inctotal", "totW"]].divide(dollars)
d = data_hs[["ctotal", "inctotal", "totW"]].divide(dollars)

a.corr(method='pearson') #no formal education
b.corr(method='pearson') # less than primary school
c.corr(method='pearson') # completed primary school
d.corr(method='pearson') #completed secondary school and higher


#%% QUESTION FOUR: CIW OVER LIFECYCLE
#=========== Preparing the dataset by converting into dollars and taking logs
import matplotlib.pyplot as plt
from numpy import inf
a = data_no[["age","ctotal", "inctotal", "totW"]]
a = a.groupby(by="age")[["ctotal", "inctotal", "totW"]].sum()
a = a[["ctotal", "inctotal", "totW"]].divide(dollars)
a = a[["ctotal", "inctotal", "totW"]].apply(np.log)
a[a == -inf] = 0
a["age"] = np.array(a.index.values)

import matplotlib.pyplot as plt
from numpy import inf
b = data_lp[["age","ctotal", "inctotal", "totW"]]
b = b.groupby(by="age")[["ctotal", "inctotal", "totW"]].sum()
b = b[["ctotal", "inctotal", "totW"]].divide(dollars)
b = b[["ctotal", "inctotal", "totW"]].apply(np.log)
b[b == -inf] = 0
b["age"] = np.array(b.index.values)

import matplotlib.pyplot as plt
from numpy import inf
c = data_p[["age","ctotal", "inctotal", "totW"]]
c = c.groupby(by="age")[["ctotal", "inctotal", "totW"]].sum()
c = c[["ctotal", "inctotal", "totW"]].divide(dollars)
c = c[["ctotal", "inctotal", "totW"]].apply(np.log)
c[c == -inf] = 0
c["age"] = np.array(c.index.values)

import matplotlib.pyplot as plt
from numpy import inf
d = data_hs[["age","ctotal", "inctotal", "totW"]]
d = d.groupby(by="age")[["ctotal", "inctotal", "totW"]].sum()
d = d[["ctotal", "inctotal", "totW"]].divide(dollars)
d = d[["ctotal", "inctotal", "totW"]].apply(np.log)
d[d == -inf] = 0
d["age"] = np.array(d.index.values)


#%% Scatter plot of Lifecycle of CIW
#**** Wealth***************
from matplotlib import pyplot
pyplot.plot(a["age"],a["totW"], label='No Educ')
pyplot.plot(b["age"],b["totW"], label='Less Pry')
pyplot.plot(c["age"],c["totW"], label='Compl.Pry')
pyplot.plot(d["age"],d["totW"], label='Sec&higher')
pyplot.legend(loc='upper right')
plt.ylabel('Wealth')
plt.xlabel('Age')
plt.title('Lifecycle Wealth of Education groups')
plt.grid(True)
pyplot.savefig('wl.png')
#******** Consumption
from matplotlib import pyplot
pyplot.plot(a["age"],a["ctotal"], label='No Educ')
pyplot.plot(b["age"],b["ctotal"], label='Less Pry')
pyplot.plot(c["age"],c["ctotal"], label='Compl.Pry')
pyplot.plot(d["age"],d["ctotal"], label='Sec&higher')
pyplot.legend(loc='upper right')
plt.ylabel('Consumption')
plt.xlabel('Age')
plt.title('Lifecycle Consumption of Education groups')
plt.grid(True)
pyplot.savefig('cl.png')
#******* Income
from matplotlib import pyplot
pyplot.plot(a["age"],a["inctotal"], label='No Educ')
pyplot.plot(b["age"],b["inctotal"], label='Less Pry')
pyplot.plot(c["age"],c["inctotal"], label='Compl.Pry')
pyplot.plot(d["age"],d["inctotal"], label='Sec&higher')
pyplot.legend(loc='upper right')
plt.ylabel('Income')
plt.xlabel('Age')
plt.title('Lifecycle Income of Education groups')
plt.grid(True)
pyplot.savefig('il.png')




#%%                         PART THREE 
####===================INEQUALITY ACROSS SPACES=============================================
#=========================================================================############################
# Q 1. Plot the level of CIW and labor supply by zone (or district) against the level of household
                                    # income by zone.
#########=========================================================================####################

import matplotlib.pyplot as plt
from numpy import inf

data_r1 = data.loc[data["region"]==1] # no formal educ
data_r2 = data.loc[data["region"]==2] # less than primary 
data_r3 = data.loc[data["region"]==3] # completed primary
data_r4 = data.loc[data["region"]==4] # completed higher than secondary

a = data_r1[["ctotal", "inctotal", "totW"]]
a = a.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
a = a[["ctotal", "inctotal", "totW"]].divide(dollars)
a = a[["ctotal", "inctotal", "totW"]].apply(np.log)
a[a == -inf] = 0
a= a.replace(np.nan,0)

b = data_r2[["ctotal", "inctotal", "totW"]]
b = b.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
b = b[["ctotal", "inctotal", "totW"]].divide(dollars)
b = b[["ctotal", "inctotal", "totW"]].apply(np.log)
b[b == -inf] = 0
b = b.replace(np.nan, 0)

c = data_r3[["ctotal", "inctotal", "totW"]]
c = c.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
c = c[["ctotal", "inctotal", "totW"]].divide(dollars)
c = c[["ctotal", "inctotal", "totW"]].apply(np.log)
c[c == -inf] = 0
c = c.replace(np.nan,0)

d = data_r4[["ctotal", "inctotal", "totW"]]
d = d.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
d = d[["ctotal", "inctotal", "totW"]].divide(dollars)
d = d[["ctotal", "inctotal", "totW"]].apply(np.log)
d[d == -inf] = 0
d = d.replace(np.nan,0)

#=========== Scatter plot
from matplotlib import pyplot
pyplot.plot(a["inctotal"],a["ctotal"],  label='Central')
pyplot.plot(b["inctotal"],b["ctotal"],label='Eastern')
pyplot.plot(c["inctotal"],c["ctotal"], label='Northern')
pyplot.plot(d["inctotal"],d["ctotal"], label='Western')
pyplot.legend(loc='upper right')
pyplot.ylabel('Log Consumption')
pyplot.xlabel('Log Income')
pyplot.title('Consumption against income by zone')
pyplot.grid(True)
pyplot.savefig('c31.png')

from matplotlib import pyplot
pyplot.plot(a["inctotal"],a["totW"], label='Central')
pyplot.plot(b["inctotal"],b["totW"], label='Eastern')
pyplot.plot(c["inctotal"],c["totW"], label='Northern')
pyplot.plot(d["inctotal"],d["totW"], label='Western')
pyplot.legend(loc='upper right')
pyplot.ylabel('Log Wealth')
pyplot.xlabel('Log Income')
pyplot.title('Wealth against income by zone')
pyplot.grid(True)
pyplot.savefig('w31.png')



############--------------------------------------------------#########################################
# Q2. Plot the inequality of CIW and labor supply by zone (or district) against the level of household
                                    #income by zone.
lab_r1 = lab.loc[lab["region"]==1] # Region 1 Central region
lab_r2 = lab.loc[lab["region"]==2] # Region 2 Eastern region
lab_r3 = lab.loc[lab["region"]==3] # Region 3 Northern region
lab_r4 = lab.loc[lab["region"]==4] # 

l1 = lab_r1[["ctotal", "inctotal", "totW"]]
#d = d.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
l1 = l1[["ctotal", "inctotal", "totW"]].divide(dollars)
l1 = l1[["ctotal", "inctotal", "totW"]].apply(np.log)
l1[l1 == -inf] = 0
l1 = l1.replace(np.nan,0)

l2 = lab_r2[["ctotal", "inctotal", "totW"]]
#d = d.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
l2 = l2[["ctotal", "inctotal", "totW"]].divide(dollars)
l2 = l2[["ctotal", "inctotal", "totW"]].apply(np.log)
l2[l2 == -inf] = 0
l2 = l2.replace(np.nan,0)

l3 = lab_r3[["ctotal", "inctotal", "totW"]]
#d = d.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
l3 = l3[["ctotal", "inctotal", "totW"]].divide(dollars)
l3 = l3[["ctotal", "inctotal", "totW"]].apply(np.log)
l3[l3 == -inf] = 0
l3 = l3.replace(np.nan,0)

l4 = lab_r4[["ctotal", "inctotal", "totW"]]
#d = d.groupby(by="inctotal")[["ctotal", "inctotal", "totW"]].sum()
l4 = l4[["ctotal", "inctotal", "totW"]].divide(dollars)
l4 = l4[["ctotal", "inctotal", "totW"]].apply(np.log)
l4[l4 == -inf] = 0
l4 = l4.replace(np.nan,0)


from matplotlib import pyplot
pyplot.scatter(l1["inctotal"],lab_r1["h"], label='Central')
pyplot.scatter(l2["inctotal"],lab_r2["h"], label='Eastern')
pyplot.scatter(l3["inctotal"],lab_r3["h"], label='Northern')
pyplot.scatter(l4["inctotal"],lab_r4["h"], label='Western')
pyplot.legend(loc='upper right')
pyplot.ylabel('Hours per worker')
pyplot.xlabel('Log Income')
pyplot.title('Labor supply against income by zone')
pyplot.grid(True)
pyplot.savefig('i21.png')

                                    
##########-----==============================================#########################################
data_r1 = data.loc[data["region"]==1] # Region 1 Central region
data_r2 = data.loc[data["region"]==2] # Region 2 Eastern region
data_r3 = data.loc[data["region"]==3] # Region 3 Northern region
data_r4 = data.loc[data["region"]==4] # Region 4 Western

####---============================================================================##########
#Here's a simple implementation of the Gini coefficient. 
#It uses the fact that the Gini coefficient is half the relative mean absolute difference.
########----------------------------------------------------------------------------###########
def gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

#======== Gini index of consumoption in regions
gini (data_r1[["ctotal"]])
gini (data_r2[["ctotal"]])
gini (data_r3[["ctotal"]])
gini (data_r4[["ctotal"]])

gini (data_r1[["inctotal"]])
gini (data_r2[["inctotal"]])
gini (data_r3[["inctotal"]])
gini (data_r4[["inctotal"]])

gini (data_r1[["totW"]])
gini (data_r2[["totW"]])
gini (data_r3[["totW"]])
gini (data_r4[["totW"]])



#------------------------------------------------------------------------

import matplotlib.pyplot as plt

def GRLC(values):
    '''
    Calculate Gini index, Gini coefficient, Robin Hood index, and points of 
    Lorenz curve based on the instructions given in 
    www.peterrosenmai.com/lorenz-curve-graphing-tool-and-gini-coefficient-calculator
    Lorenz curve values as given as lists of x & y points [[x1, x2], [y1, y2]]
    @param values: List of values
    @return: [Gini index, Gini coefficient, Robin Hood index, [Lorenz curve]] 
    '''
    n = len(values)
    assert(n > 0), 'Empty list of values'
    sortedValues = sorted(values) #Sort smallest to largest

    #Find cumulative totals
    cumm = [0]
    for i in range(n):
        cumm.append(sum(sortedValues[0:(i + 1)]))

    #Calculate Lorenz points
    LorenzPoints = [[], []]
    sumYs = 0           #Some of all y values
    robinHoodIdx = -1   #Robin Hood index max(x_i, y_i)
    for i in range(1, n + 2):
        x = 100.0 * (i - 1)/n
        y = 100.0 * (cumm[i - 1]/float(cumm[n]))
        LorenzPoints[0].append(x)
        LorenzPoints[1].append(y)
        sumYs += y
        maxX_Y = x - y
        if maxX_Y > robinHoodIdx: robinHoodIdx = maxX_Y   
    
    giniIdx = 100 + (100 - 2 * sumYs)/n #Gini index 

    return [giniIdx, giniIdx/100, robinHoodIdx, LorenzPoints]

## ------- Lorenz curve and Gini for Consumption
a1 = data_r1["ctotal"].divide(dollars)
a2 = data_r2["ctotal"].divide(dollars)
a3 = data_r3["ctotal"].divide(dollars)
a4 = data_r4["ctotal"].divide(dollars)

result = GRLC(a1)
result2 = GRLC(a2)
result3 = GRLC(a3)
result4 = GRLC(a4)

print ('Gini Index', result[0]  )
print ('Gini Coefficient', result[1])
print ('Robin Hood Index', result[2])
print ('Lorenz curve points', result[3])

import matplotlib.pyplot as plt
plt.plot([0, 100], [0, 100],'--',color='k')
plt.plot(result[3][0], result[3][1], label='Central' )
plt.plot(result2[3][0], result2[3][1], label='Eastern')
plt.plot(result3[3][0], result3[3][1], label ='Northern')
plt.plot(result4[3][0], result4[3][1], label = 'Western')
plt.xlabel('% of population')
plt.ylabel('% of values')
plt.title('Lorenz curve of Consumption by zone')
plt.legend(loc='upper right')
plt.savefig('c5.png')
plt.show()

del result, result2, result3, result4

###====== Lorenz curve for income distribution 
b1 = np.ceil(data_r1["inctotal"].divide(dollars))
b2 = np.ceil(data_r2["inctotal"].divide(dollars))
b3 = np.ceil(data_r3["inctotal"].divide(dollars))
b4 = np.ceil(data_r4["inctotal"].divide(dollars))

result = GRLC(b1)
result2 = GRLC(b2)
result3 = GRLC(b3)
result4 = GRLC(b4)

print ('Gini Index', result[0]  )
print ('Gini Coefficient', result[1])
print ('Robin Hood Index', result[2])
print ('Lorenz curve points', result[3])

import matplotlib.pyplot as plt
plt.plot([0, 100], [0, 100],'--',color='k')
plt.plot(result[3][0], result[3][1], label='Central' )
plt.plot(result2[3][0], result2[3][1], label='Eastern')
plt.plot(result3[3][0], result3[3][1], label ='Northern')
plt.plot(result4[3][0], result4[3][1], label = 'Western')
plt.xlabel('% of population')
plt.ylabel('% of values')
plt.title('Lorenz curve of Income by zone')
plt.legend(loc='upper right')
plt.savefig('i5.png')
plt.show()
del result, result2, result3, result4

#-------------------- Wealth distribution by zone
c1 = data_r1["totW"].divide(dollars)
c2 = data_r2["totW"].divide(dollars)
c3 = data_r3["totW"].divide(dollars)
c4 = data_r4["totW"].divide(dollars)

result = GRLC(c1)
result2 = GRLC(c2)
result3 = GRLC(c3)
result4 = GRLC(c4)

print ('Gini Index', result[0]  )
print ('Gini Coefficient', result[1])
print ('Robin Hood Index', result[2])
print ('Lorenz curve points', result[3])

import matplotlib.pyplot as plt
plt.plot([0, 100], [0, 100],'--',color='k')
plt.plot(result[3][0], result[3][1], label='Central' )
plt.plot(result2[3][0], result2[3][1], label='Eastern')
plt.plot(result3[3][0], result3[3][1], label ='Northern')
plt.plot(result4[3][0], result4[3][1], label = 'Western')
plt.xlabel('% of population')
plt.ylabel('% of values')
plt.title('Lorenz curve of Wealth by zone')
plt.legend(loc='upper right')
plt.savefig('w5.png')
plt.show()
del result, result2, result3, result4



#######==================---------------------------------------#####################################
## 3. Plot the covariances of CIW and labor supply by zone (or district) against the level of household
#============================income by zone.=====================================================
import seaborn as sns

e1 = data_r1[["ctotal","inctotal","totW"]]/dollars
# calculate the correlation matrix
corr = e1.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns) # plot the heatmap
plt.title("Covariance matrix of Central region")
plt.savefig('cov1')

#------- Income
e3 = data_r3[["ctotal","inctotal","totW"]]/dollars
# calculate the correlation matrix
corr = e3.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns) # plot the heatmap
plt.title("Covariance matrix of Nothern region")
plt.savefig('cov3.png')

#======= Wealth
e4 = data_r4[["ctotal","inctotal","totW"]]/dollars
# calculate the correlation matrix
corr = e4.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns) # plot the heatmap
plt.title("Covariance matrix of Western region")
plt.savefig('cov4.png')

######=================================================================###########################
## 4. Reproduce the Bick et. al (2018) analysis between individual vs. country income, and individual
#====    hours-wage elasticity by country in Uganda. Instead of "country" use districts/zones
##=============================# in Uganda. #####################################################
#
import statsmodels.formula.api as sm

#Create dummies
dummies = pd.get_dummies(lab['region'])
dummies.columns = ["region1","region2","region3","region4"]
dummies.drop(["region1"], axis=1, inplace=True)
# 1:central, 2:Eastern, 3:Northern, 4:Western
lab = lab.join(dummies)
dummies = pd.get_dummies(lab['sex'])
dummies.columns = ["male","female"]
dummies.drop(["male"], axis=1, inplace=True)
lab = lab.join(dummies)

 #Generate logs
for item in ['ctotal', 'ctotal_dur', 'ctotal_gift', 'totalh','pay','inctotal','cfood', 'cnodur', 'wage_total', 'bs_profit', 'profit_agr', 'profit_ls', 'total_agrls']:
    lab["ln"+item] = np.log(lab[item]+np.abs(np.min(lab[item])))
    #data["ln"+item] = np.log(data[item])
    
    
lab["lntotalh"][lab["lntotalh"] == -inf] = 0
lab["lntotalh"] = lab["lntotalh"].replace(np.nan,0)
lab["lnpay"][lab["lnpay"] == -inf] = 0
lab["lnpay"] = lab["lnpay"].replace(np.nan,0)
lab["age_sq"] = lab["age"]*lab["age"]
olsc = sm.ols(formula="lntotalh ~ lnpay+age +age_sq +region2 +region3 +region4 ", data=lab).fit()
print(olsc.summary())


