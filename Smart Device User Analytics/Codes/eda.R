library(tidyverse)

# Daily activities

dailyActivity <- read_csv("../data/Daily/dailyActivity.csv")

dailyActivity %>% 
  filter(TotalSteps>100) %>% 
  ggplot() +
  geom_point(aes(x=TotalSteps, y=Calories), color="cornflowerblue") +
  labs(title="Daily Calories vs Total Steps")

df <- data.frame(matrix(ncol=3))
colnames(df) <- c("Id", "Measurement", "Type")
for (i in 1:nrow(dailyActivity)) {
  tuple <- list(dailyActivity$Id[i], dailyActivity$VeryActiveMinutes[i], "VeryActiveMinutes")
  df <- rbind(df, tuple)
  tuple <- list(dailyActivity$Id[i], dailyActivity$LightlyActiveMinutes[i], "LightlyActiveMinutes")
  df <- rbind(df, tuple)
  tuple <- list(dailyActivity$Id[i], dailyActivity$FairlyActiveMinutes[i], "FairlyActiveMinutes")
  df <- rbind(df, tuple)
  tuple <- list(dailyActivity$Id[i], dailyActivity$SedentaryMinutes[i], "SedentaryMinutes")
  df <- rbind(df, tuple)
}
df <- drop_na(df)

id <- df$Id[1]
y <- 1
for (i in 1:nrow(df)){
  id2 <- df$Id[i]
  if (id2==id){
    df$idx[i] = y
  }
  else{
    id <- df$Id[i]
    y <- y+1
    df$idx[i] =y
  }
}

df %>% 
  group_by(idx, Type) %>% 
  summarize(meas=mean(Measurement)) %>%
  ggplot() +
    geom_point(aes(x=idx, y=meas, color=Type), show.legend = FALSE) +
    labs(title="Minutes of Activity Instensities", x="Users", y="Minutes") +
    facet_wrap(~Type)


# Daily sleep

sleepDay <- read_csv("../data/Daily/sleepDay.csv")

sleepDay %>% 
  group_by(Id) %>% 
  summarize(sleep=mean(TotalHoursAsleep)) %>%
  filter(sleep>3) %>% 
  mutate(idx = row_number()) %>% 
  ggplot() +
  geom_bar(aes(idx,sleep),stat='identity') +
  geom_hline(yintercept=7, size=1) +
  annotate("text", x=18, y=9, label="Average: 7 Hours", color="orange", size=5) +
  labs(title="Average Sleep Time", x="Users", y="Hours")

sleepDay %>% 
  group_by(Id) %>% 
  summarize(sleep=mean(TotalMinutesInBed-TotalMinutesAsleep)) %>%
  filter(sleep<300) %>% 
  mutate(idx = row_number()) %>% 
  ggplot() +
  geom_bar(aes(idx,sleep),stat='identity') +
  labs(title="Bed Time Without Sleeping", x="Users", y="Minutes") 


# Heart rates

heartrate_seconds <- read_csv("../data/heartrate_seconds.csv")

heartrate_seconds %>% 
  group_by(Id) %>% 
  summarize(mean=mean(Value)) %>% 
  mutate(idx = row_number()) %>%
  ggplot()+
    geom_bar(aes(idx,mean),stat="identity")+
  ylim(0, 105)+
  geom_hline(yintercept=80, size=1,) +
  geom_label(aes(x = 8, y = 100, label = "Average: 80 bpm"), size=5, fill = "white")+
  geom_label(aes(x = 2.5, y = 23, label = "Max: 94 bpm"), fill = "white")+
  geom_label(aes(x = 2.5, y = 13, label = "Min: 66 bpm"), fill = "white")+
  labs(title="Heartrates by Users",
       x="User", y="Heartbeat (per minute)")


# Weight and BMI

weightLogInfo <- read_csv("../data/weightLogInfo.csv")

weightLogInfo %>% 
  group_by(Id) %>% 
  summarize(mean=mean(BMI)) %>% 
  filter(mean<40) %>% 
  mutate(idx = row_number()) %>%
  ggplot()+
  geom_bar(aes(idx,mean),stat="identity")+
  ylim(0, 35)+
  geom_hline(yintercept=25.2, size=1,) +
  geom_label(aes(x = 5, y = 32, label = "Average: 25.2"), size=5, fill = "white") +
  labs(title="BMI by Users",
       x="User", y="BMI")
