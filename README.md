# Growth and Development
Growth and Development Class, Winter 2019, IDEA PhD, UAB.

# Order of Codes, Results and Data
Software used: Python
To begin with, the codes are structured in a manner to make interpretation easier and to enhance reproducibility. The codes are structured as follows:

1. For agricultural production, profit and crop production is titled 'agric'
2. For consumption including livestock and others is titled 'csp'
3. For labor wages, business profit and transfers is titled 'labor'
4. For social demography / background/descriptive statistics of respondents is titled 'socdem'
5. For merging of the above 1 - 4 dataset is titled 'fullanalysis'
6. For the problem set 1 questions is titled 'PS1'
7. The results, output ad intepretation of charts, tables and others of the problem set is titled # BabaniyiOlaniyiFullPS1

# Data Description:
The LSMS-ISA project is supporting the design and implementation of the Uganda National Panel Survey (UNPS), with a focus on expanding the agricultural content of the UNPS as well as ensuring comparability with other surveys being carried out under the LSMS-ISA project in Sub-Saharan Africa. The emphasis is to ensure that information on agriculture and livestock, and data on food and nutrition security inter alia, are mainstreamed into the UNPS, and that the quality and relevance of these data is further improved and made sustainable over time. For this problem set, we use the household and agriculture data set collected from 2013 – 2014 for Uganda.

# Sampling and Survey Design
The UNPS sample is approximately 3,200 households, all of whom had been previously interviewed as part of the 2005/2006 Uganda National Household Survey (UNHS). The sample also includes a randomly-selected share of split-off households that were formed after the 2005/2006 UNHS. The UNPS is representative at the national, urban/rural and main regional levels (North, East, West and Central regions). For the agriculture dataset, areas of all owned and/or cultivated agricultural plots were measured via GPS and farmer-supplied area estimates have been validated. Crop cards were used to better quantify the production of continuously harvested as well as staple crops.

# Trimming strategy
Level-trimming: trim top and bottom 1% of all components of income, transfers and consumption by residency then construct household aggregates, then trim top and bottom 1% of the household aggregates.

# Correcting inflation
For this dataset, I did not correct for inflation, however, I converted the Ugandan shillings to its dollar equivalent of 2586.89 according to 2013 exchange rate.
# Definition of variables
My consumption measure includes: Food consumption, Utilities (from subsidies), Housing Service: rent, Childcare, Medical care, Semi-durable Supplies: appliances, electronic devices, kitchenware, durable and non-durable foods, gifts.
My wealth measure includes: Estimated value of assets such as land equipment.etc.
My income measure includes: Income from agriculture such as livestock and crops, labor, business, transfers, gifts.
