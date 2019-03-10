import pandas as pd
import numpy as np
# =============================================================================
#  Merging Sociodemographic characteristics
# =============================================================================

#Region
region = pd.read_stata('GSEC1.dta', convert_categoricals=False)
region = region[["HHID","region"]]
region.columns =["hh","region"]
region.region = pd.to_numeric(region.region)

#Age
age = pd.read_stata('GSEC2.dta', convert_categoricals=False)
age = age[["PID","h2q3","h2q4","h2q8"]]
age.columns = ["pid","sex","hh_member", "age"]


## Health
health = pd.read_stata('GSEC5.dta', convert_categoricals=False)
health = health[["PID","h5q5"]]
health.columns = ["pid","illdays"]


#Background
bck = pd.read_stata('GSEC3.dta', convert_categoricals=False)
bck = bck[["PID","h2q1","h3q3","h3q4", "h3q6","h3q7"]]
bck.columns = ["pid","hhnum","father_educ", "father_ocup","mother_educ","mother_ocup"]
bck.father_educ = bck.father_educ.replace(99,np.nan)

#Education
educ = pd.read_stata('GSEC4.dta', convert_categoricals=False)
educ = educ[["HHID","PID","h4q4", "h4q7"]]
educ.columns = ["hh","pid","writeread","classeduc"]
educ.writeread = educ.writeread.replace([2, 4, 5],0)
educ.loc[educ["classeduc"]==99, "classeduc"] = np.nan
#1 if able to read and write. 0 if unable both, unable writing, uses braille

#Sociodemographic dataset
socio = pd.merge(age, bck, on="pid", how="inner")
socio = pd.merge(socio, educ, on="pid", how="inner")
socio = pd.merge(socio, health, on="pid", how="inner")
socio = socio.loc[(socio.hh_member==1)]
socio.drop(["hh_member", "pid"], axis=1, inplace=True)
#socio.drop(socio.columns[0], axis=1, inplace=True)


socio = pd.merge(socio, region, on="hh", how="inner")
socio.to_csv("sociodem.csv")
