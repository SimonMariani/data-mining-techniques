# Data Mining Techniques
# Assignment 1
######################################################################################
setwd("D:/1. school/Semester 2/4. Data Mining/2. assignments/assignment 1/data-mining-techniques/data_processed")
library(stargazer)
library(ggplot2)
library(gridExtra)
library(qqplotr)
library(cluster)
library(factoextra)
library(corrplot)
library(gridExtra)
######################################################################################
data <- read.csv("subdata_pr_su.csv")
data <- data[,c(1:21)]

#participants.index <- which(data$X %in% 0)
#participants.amount <- length(days.index)

#for(i in 1:participants.amount) {
#    print(paste0("PARTICIPANT ", i))
#    participant.data <- data[participants.index[i]:(participants.index[i+1]-1),]
#    print(summary(participant.data))
#}
attach(data)
var <- cbind(mood, circumplex.arousal, circumplex.valence, activity, screen, call, sms, appCat.builtin, appCat.communication, appCat.entertainment, appCat.finance, appCat.game, appCat.office, appCat.other, appCat.social, appCat.travel, appCat.unknown, appCat.utilities, appCat.weather)
aggregated.data <- aggregate(var ~ X, data, mean)[-1]

# Descriptive Stats
summary(aggregated.data)
stargazer(aggregated.data)

# Distributions
# Density
for (i in 1:19) {
  p <- ggplot(aggregated.data, aes(x=aggregated.data[,i])) + 
    geom_density(fill="blue", alpha=0.4) +
    xlab(colnames(aggregated.data)[i])
  print(p)
}

figure <- grid.arrange(hist1,hist2,ncol=2)

# QQplots
for (i in 1:19) {
  p <- ggplot(aggregated.data, aes(sample=aggregated.data[,i])) + 
    geom_qq_band(bandType = "pointwise", mapping = aes(fill = "Normal"), alpha = 0.5, fill="#00c7ca") +
    stat_qq_line() +
    stat_qq_point() +
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") +
    ggtitle(colnames(aggregated.data)[i]) + 
    scale_fill_discrete("Bandtype") 
  print(p)
}

# K-means
cluster.data <- read.csv("aggregated_participants.csv")[-1]

clustering1 <- kmeans(cluster.data, centers=2)
clustering2 <- kmeans(cluster.data, centers=3)
clustering3 <- kmeans(cluster.data, centers=4)
clustering4 <- kmeans(cluster.data, centers=5)

p1 <- fviz_cluster(clustering1, geom = "point", data = cluster.data) + ggtitle("k = 2")
p2 <- fviz_cluster(clustering2, geom = "point",  data = cluster.data) + ggtitle("k = 3")
p3 <- fviz_cluster(clustering3, geom = "point",  data = cluster.data) + ggtitle("k = 4")
p4 <- fviz_cluster(clustering4, geom = "point",  data = cluster.data) + ggtitle("k = 5")
grid.arrange(p1, p2, p3, p4, nrow = 2)

fviz_nbclust(cluster.data, kmeans, method = "silhouette")

cluster.data["Cluster"] <- clustering1$cluster
cluster1 <- cluster.data[cluster.data$Cluster == 1,]
cluster2 <- cluster.data[cluster.data$Cluster == 2,]

# Densities
vars <- colnames(data)[-c(1,2,3, 8, 9)]
for (i in 2:17) {
  p <- ggplot(cluster.data, aes(x=cluster.data[,i], y=X0, color=as.factor(Cluster))) + 
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = vars[i-1], y = "Mood") +
    scale_colour_manual(values=c("red", "blue"), name = "Cluster") 
  print(p)
}

# Significance:
formula <- X0 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14 + X15 + X16
summary(lm(formula, data=cluster.data))
stargazer(lm(formula, data=cluster.data))
summary(lm(X0 ~ X3 + X9 + X11, data=cluster.data))
stargazer(lm(X0 ~ X3 + X9 + X11, data=cluster.data))

# Correlation plot
colnames(cluster.data) <- c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q")
corrplot(cor(cluster.data), method="color", type="upper", tl.col="black", tl.srt=45)
corrplot(cor(cluster.data), method="number", type="upper", tl.col="black", tl.srt=45)
corrplot(cor(cluster.data), method="pie", type="upper", tl.col="black", tl.srt=45)
