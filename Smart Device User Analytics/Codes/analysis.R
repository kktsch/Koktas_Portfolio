library(tidyverse)
library(psych)

# Data prep for daily activities

dailyActivity <- read_csv("../data/Daily/dailyActivity.csv")

sleepDay <- read_csv("../data/Daily/sleepDay.csv", 
                     col_select=list("Id","SleepDay","TotalHoursAsleep","TotalHoursInBed"))
sleepDay <- rename(sleepDay, ActivityDate = SleepDay)
sleepDay$TotalHoursQuietSleep <- sleepDay$TotalHoursInBed-sleepDay$TotalHoursAsleep
sleepDay <- sleepDay[-4]

df <- merge(dailyActivity,sleepDay)


# Correlation analysis

df <- df[-c(4:7)]
pairs.panels(df[,c(3:6)], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)


# Regression for total steps and calories relationship

regression_model <- lm(Calories ~ TotalSteps, data=df)


# Bar chart for wakefulness time in bed

df %>% 
  mutate(Day=weekdays(as.Date(ActivityDate, format="%m/%d/%y"))) %>% 
  group_by(Day) %>% 
  summarize(mean=mean(TotalHoursQuietSleep)) %>% 
  ggplot()+
  geom_bar(aes(Day,mean),stat="identity")+
  labs(title="Average Wakefulness Hours by Days",x="Days",y="Hours")

# Significancy test
df %>% 
  mutate(Day=weekdays(as.Date(ActivityDate, format="%m/%d/%y"))) %>% 
  select(TotalHoursQuietSleep,Day) %>% 
  filter(Day=="Cuma") %>% 
  select(TotalHoursQuietSleep) -> friday
df %>% 
  mutate(Day=weekdays(as.Date(ActivityDate, format="%m/%d/%y"))) %>% 
  select(TotalHoursQuietSleep,Day) %>% 
  filter(Day!="Cuma") %>% 
  select(TotalHoursQuietSleep) -> others

t.test(friday, y = others, alternative = "greater")
  




