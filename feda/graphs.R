library(dplyr)
library(ggplot2)

summary_FEDA_test <- read.csv("~/PycharmProjects/Project2_SML/feda/summary_FEDA_test_ratio.csv")
summary_BOOST_test <- read.csv("~/PycharmProjects/Project2_SML/TrAdaBoostin_Regressor/summary_BOOST_test_ratio.csv")

ggplot(summary_FEDA_test, aes(x = samples, y = mse, color = model)) +
  geom_line(size=1) +
  facet_wrap(~target, nrow=3) +
  theme_classic() +
  theme(text = element_text(size=30)) 

dataset <- read.csv("~/PycharmProjects/Project2_SML/feda/dataset.csv")



ggplot(summary_BOOST_test, aes(x = samples, y = mse, color = model)) +
  geom_line() +
  facet_wrap(~target, nrow=2) +
  theme_classic()
