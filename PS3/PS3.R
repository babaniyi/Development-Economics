rm(list=ls())
data = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/datadiff.csv",header=T)
names(data) = c("hh","wave","dc","dy","dcb")
data = na.exclude(data)
out <- lapply(unique(data$hh), function(s){
  sub_iris <- subset(data, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})
out <- do.call(rbind, out)
out
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
hist(out$dcb)
mean(out$dy)
mean(out$dcb)
median(out$dy)
median(out$dcb)

library(ggplot2)
ggplot(data=out, aes(out$dy)) + 
  geom_histogram(breaks=seq(-7, 7, by =1), 
                 col="red", 
                 aes(fill=..count..))+
labs(title="Histogram of Betas") +
  labs(x="Beta", y="Count")

ggplot(data=out, aes(out$dcb)) + 
  geom_histogram(breaks=seq(-15, 15, by =2), 
                 col="red", 
                 aes(fill=..count..)) +
  scale_fill_gradient("Count", low = "green", high = "red")+
labs(title="Histogram of Phis'") +
  labs(x="Beta", y="Count")


#*****************************************************************************************#
#***************** QUESTION TWO ****************************************
##############################################################################

#-------------------------------------------------------------
rm(list=ls())
dat0 = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/dat0.csv",header=T)
dat0 = na.exclude(dat0)

out <- lapply(unique(dat0$hh), function(s){
  sub_iris <- subset(dat0, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})

out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
mean(out$dy)
median(out$dy)

library(ggplot2)
ggplot(data=out, aes(out$dy)) + 
  geom_histogram(breaks=seq(-10, 10, by =2), 
                 col="red", 
                 aes(fill=..count..))+
  labs(title="Histogram of Betas for Bottom 20%") +
  labs(x="Beta", y="Count")


#---------------------------------------------------------------######
# Bottom 20-40%
#####################################################################
rm(list=ls())
dat1 = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/dat1.csv",header=T)
dat1 = na.exclude(dat1)

out <- lapply(unique(dat1$hh), function(s){
  sub_iris <- subset(dat1, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})

out <- do.call(rbind, out)
out
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
mean(out$dy)
median(out$dy)

###################################################################################
##---- 40-60% 
rm(list=ls())
dat0 = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/dat2.csv",header=T)
dat0 = na.exclude(dat0)

out <- lapply(unique(dat0$hh), function(s){
  sub_iris <- subset(dat0, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})

out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
mean(out$dy)
median(out$dy)

library(ggplot2)
ggplot(data=out, aes(out$dy)) + 
  geom_histogram(breaks=seq(-10, 10, by =5), 
                 col="red", 
                 aes(fill=..count..))+
  labs(title="Histogram of Betas for Bottom 20%") +
  labs(x="Beta", y="Count")


##################################################################################
##---- 60-80% 
rm(list=ls())
dat0 = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/dat3.csv",header=T)
dat0 = na.exclude(dat0)

out <- lapply(unique(dat0$hh), function(s){
  sub_iris <- subset(dat0, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})

out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
mean(out$dy)
median(out$dy)


##################################################################################
##---- 80-100% 
rm(list=ls())
dat4 = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/dat4.csv",header=T)
dat4 = na.exclude(dat4)

out <- lapply(unique(dat4$hh), function(s){
  sub_iris <- subset(dat4, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})

out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
mean(out$dy)
median(out$dy)


library(ggplot2)
ggplot(data=out, aes(out$dy)) + 
  geom_histogram(breaks=seq(-10, 10, by =1), 
                 col="red", 
                 aes(fill=..count..))+
  labs(title="Histogram of Betas for Top 20%") +
  labs(x="Beta", y="Count")
#-------------------------------------------------------------------------
  
#*******************************************************************************
#**************************QUESTION TWO B**********
rm(list=ls())
dat0 = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Project/project_data/dat3.csv",header=T)
dat0 = na.exclude(dat0)

out <- lapply(unique(dat0$hh), function(s){
  sub_iris <- subset(dat0, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})

out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
mean(out$dy)
median(out$dy)


#**** QUESTION FOUR****
rm(list=ls())
data = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Development/Solutions/Thesis-Heterogeneities-Risk-Insurance-Uganda-master/datadiff_r.csv",header=T)
names(data) = c("hh","wave","dc","dy","dcb")
data = na.exclude(data)
out <- lapply(unique(data$hh), function(s){
  sub_iris <- subset(data, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})
out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
hist(out$dcb)
mean(out$dy)
mean(out$dcb)
median(out$dy)
median(out$dcb)

library(ggplot2)
ggplot(data=out, aes(out$dy)) + 
  geom_histogram(breaks=seq(-7, 7, by =1), 
                 col="red", 
                 aes(fill=..count..))+
  labs(title="Histogram of Betas of Rural households") +
  labs(x="Beta", y="Count")

ggplot(data=out, aes(out$dcb)) + 
  geom_histogram(breaks=seq(-7, 7, by =1), 
                 col="red", 
                 aes(fill=..count..)) +
  scale_fill_gradient("Count", low = "green", high = "red")+
  labs(title="Histogram of Phi of Rural household") +
  labs(x="Beta", y="Count")

#### URBAN Households
rm(list=ls())
data = read.csv("C:/Users/User/Documents/IDEA/Year2/SemesterTwo/Development/Solutions/Thesis-Heterogeneities-Risk-Insurance-Uganda-master/datadiff_u.csv",header=T)
names(data) = c("hh","wave","dc","dy","dcb")
data = na.exclude(data)
out <- lapply(unique(data$hh), function(s){
  sub_iris <- subset(data, hh == s)
  m <- lm(dc ~ dy+dcb, data = sub_iris)
  coef(m)
})
out <- do.call(rbind, out)
out = na.exclude(out)
out = as.data.frame(out)
hist(out$dy)
hist(out$dcb)
mean(out$dy)
mean(out$dcb)
median(out$dy)
median(out$dcb)

library(ggplot2)
ggplot(data=out, aes(out$dy)) + 
  geom_histogram(breaks=seq(-7,7, by =1), 
                 col="red", 
                 aes(fill=..count..))+
  labs(title="Histogram of Betas of Urban households") +
  labs(x="Beta", y="Count")

ggplot(data=out, aes(out$dcb)) + 
  geom_histogram(breaks=seq(-20, 20, by =5), 
                 col="red", 
                 aes(fill=..count..)) +
  scale_fill_gradient("Count", low = "green", high = "red")+
  labs(title="Histogram of Phi of Urban household") +
  labs(x="Beta", y="Count")