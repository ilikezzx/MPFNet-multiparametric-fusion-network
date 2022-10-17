###############################################################
#   这个写关于特征筛选 + cox生存分析
#   两种特征选择方式，随机森林(根据重要性排序)+Lasso去零的特征
###############################################################


# install.packages('timeROC')
# install.packages('ggDCA')
# install.packages('VIM')
# install.packages('pec')
# install.packages('survminer')
install.packages('ggprism')

# devtools::install_github('yikeshu0611/ggDCA')

rm(list=ls()) # 清空数据
setwd("C:\\Users\\12828\\Desktop\\osteosarcoma\\os_survival")
print(getwd())


library(MASS)
library(glmnet)
library(car)
library(rms)   ###加载rms包#
library(pROC)
library(rmda)
library(xgboost)      # boost向量
library(randomForest) # 随机森林
library(timeROC)
library(ggDCA)#R语言4.1版本
library(VIM) ## 包中aggr()函数，判断数据缺失情况
library(pec)
library(survival)
library(survminer)
library(ggprism)

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

# units(data$time) <- "month"

######################      读取数据     ####################
# 读取训练集
training_dataset_path <- choose.files(caption = "请选训练集数据文件的csv文件。",
                                      multi = TRUE, filters = Filters,
                                      index = nrow(Filters))
#读取数据
training_dataset<- read.csv(training_dataset_path, header = TRUE,sep=",", stringsAsFactors=FALSE)
train_img_features <- data.matrix(training_dataset[, 1:851])
train_clin_features <- data.frame(training_dataset[, 852:856])
train_label <- training_dataset[, 857:858]
#查看数据
print(paste0("该训练集有 ",dim(training_dataset)[1]," 个样本；",dim(training_dataset)[2]," 个变量"))


# 测试集
validation_dataset_path <- choose.files(caption = "请选训练集数据文件的txt文件。",
                                        multi = TRUE, filters = Filters,
                                        index = nrow(Filters))
# 读取数据
validation_dataset<- read.csv(validation_dataset_path, header = TRUE,sep=",", stringsAsFactors=FALSE)
val_img_features <- data.matrix(validation_dataset[, 1:851])
val_clin_features <- data.frame(validation_dataset[, 852:856])
val_label <- validation_dataset[, 857:858]
# 查看数据
print(paste0("该验证集有 ",dim(validation_dataset)[1]," 个样本；",dim(validation_dataset)[2]," 个变量"))

aggr(train_label,prop=T,numbers=T) #判断数据缺失情况，红色表示有缺失。
aggr(val_label,prop=T,numbers=T) #判断数据缺失情况，红色表示有缺失。

######################      数据整理     ####################
#用for循环语句将数值型变量转为因子变量
# for(i in names(train_clin_features)[c(2:4)]) {train_clin_features[,i] <- as.factor(train_clin_features[,i])}
# for(i in names(val_clin_features)[c(2:4)]) {val_clin_features[,i] <- as.factor(val_clin_features[,i])}

train_clin_features$sex <- factor(train_clin_features$sex,levels = c(1,0),labels = c("male", "female"))
train_clin_features$lung_metastases <- factor(train_clin_features$lung_metastases,levels = c(1,0),labels = c("metastases", "local"))
# train_clin_features$tumor.stage <- factor(train_clin_features$tumor.stage,levels = c(4,3,2,1),labels = c("G4", "G3","G2","G1"))
val_clin_features$sex <- factor(val_clin_features$sex,levels = c(1,0),labels = c("male", "female"))
val_clin_features$lung_metastases <- factor(val_clin_features$lung_metastases,levels = c(1,0),labels = c("metastases", "local"))
# val_clin_features$tumor.stage <- factor(val_clin_features$tumor.stage,levels = c(4,3,2,1),labels = c("G4", "G3","G2","G1"))
train_clin_features$volume <- train_clin_features$volume / 1000
val_clin_features$volume <- val_clin_features$volume / 1000
colnames(train_clin_features)[5] <- 'stage'
colnames(val_clin_features)[5] <- 'stage'


x <- data.matrix(train_img_features)
y <- data.matrix(Surv(train_label$os_time, train_label$censor==0))


######################      随机森林多因素分析     ####################
random_train_data <- data.frame(cbind(train_img_features, train_label$censor))
colnames(random_train_data)[length(random_train_data)] <- 'survival'

random_val_data <- data.frame(cbind(val_img_features, val_label$censor))
colnames(random_val_data)[length(random_val_data)] <- 'survival'

gx.rf<-randomForest(survival~., data=random_train_data, importance=TRUE, replace=TRUE, ntree=5000)
print(gx.rf)
importance(gx.rf)
varImpPlot(gx.rf)

#使用训练集，查看预测精度
train_pred <- data.matrix(predict(gx.rf, random_train_data))
plot(random_train_data$survival, train_pred, main = '训练集', xlab = 'true five_os', ylab = 'Predict')
abline(0, 1)
train_modelroc <- roc(random_train_data$survival, train_pred)
plot(train_modelroc, print.auc=TRUE, auc.polygon=TRUE)


#使用测试集，评估预测性能
val_pred <- predict(gx.rf, random_val_data)
plot(random_val_data$survival, val_pred, main = '测试集', xlab = 'true five_os', ylab = 'Predict')
abline(0, 1)
val_modelroc <- roc(random_val_data$survival, val_pred)
plot(val_modelroc, print.auc=TRUE, auc.polygon=TRUE)



######################      logistic多因素分析     ####################
#调用glmnet包中的glmnet函数，注意family那里一定要制定是“cox”，如果是做logistic需要换成"binomial"。
fit <-glmnet(x,y,family = "cox",alpha = 1)
plot(fit,xvar="lambda",label=T)

#主要在做交叉验证,lasso
set.seed(555)
fitcv <- cv.glmnet(x,y,family="cox",alpha=1, type.measure='C', nfolds=5)
plot(fitcv)
coeff <- coef(fitcv, s="lambda.min")

active_index <- which(as.numeric(coeff)!=0)
active_coeff <- as.numeric(coeff)[active_index]
cox_name <- row.names(coeff)[active_index]
img_length <- length(cox_name)
print(cox_name)
print(img_length)

# 纯影像学训练集的效果判断
train_pred <- data.matrix(predict(fit, train_img_features, s=fitcv$lambda.1se, type="link"))
colnames(train_pred)[1] <- 'imaging_score'

# 纯影像学测试集的效果判断
val_pred <- data.matrix(predict(fit, val_img_features, s=fitcv$lambda.1se, type="link"))
colnames(val_pred)[1] <- 'imaging_score'


######################      cox分析     ####################
new_train <- data.frame(cbind(train_pred, train_clin_features, train_label))
colnames(new_train)[1] <- 'imaging_score'
new_val <- data.frame(cbind(val_pred, val_clin_features, val_label))
colnames(new_val)[1] <- 'imaging_score'
 
new_train <- data.frame(rbind(new_train, new_train,new_train,new_train))
new_val <- data.frame(rbind(new_val, new_val))


dd<-datadist(new_train) #设置工作环境变量，将数据整合
options(datadist='dd') #设置工作环境变量，将数据整合

inc <-36

#拟合cox回归
coxm <- cph(Surv(os_time,censor==0)~ imaging_score+stage+volume+sex+age,x=T,y=T,data=new_train,surv=T, time.inc = inc) 
cox.zph(coxm)#等比例风险假定
print(coxm)

# 计算C-index 数值 
train.c_index <- rcorrcens(Surv(os_time,censor==0) ~ predict(coxm), data = new_train)
print(paste('训练集c-index指数:',1-train.c_index[1]))
val.c_index <- rcorrcens(Surv(new_val$os_time,new_val$censor==0) ~ predict(coxm, new_val), data = new_val)
print(paste('测试集c-index指数:',1-val.c_index[1]))

surv <- Survival(coxm) # 建立生存函数
med  <- Quantile(coxm)

surv1 <- function(x) surv(12*1,lp=x) # 定义time.inc,3月OS
surv2 <- function(x) surv(12*3,lp=x) # 定义time.inc,6月OS
surv3 <- function(x) surv(12*5,lp=x) # 定义time.inc,1年OS



nom <- nomogram(coxm, fun=list(surv1, surv2, surv3),
                funlabel=c('1-year Survival','3-year survival','5-year survival'),
                fun.at=c('0.99','0.85','0.5','0.1'))

plot(nom)


# f<-coxph(Surv(os_time,censor==0)~imaging_score+lung_metastases+volume+sex,data=new_train)
# sum.surv<-summary(f)
# print(sum.surv)
# c_index<-sum.surv$concordance
# c_index  ## 模型区分度

# 
# f<-coxph(Surv(os_time,censor==0)~ imaging_score+volume+sex+age+lung_metastases,data=new_val)
# sum.surv<-summary(f)
# print(sum.surv)
# c_index<-sum.surv$concordance
# c_index  ## 模型区分度

# 训练集DCA曲线
train.dca <- ggDCA::dca(coxm,times = inc)
ggplot(train.dca, smooth = T, model.names="模型1",
       linetype =F, #线型
       lwd = 1.2)

# 验证集DCA曲线
val.dca <- ggDCA::dca(coxm,times = inc, new.data=new_val)
ggplot(val.dca, smooth = T)

# 训练集校准曲线
train.cal <- calibrate(coxm, u=inc, method = "boot",cmethod = 'KM',m=75, B=1000)
par(mar=c(7,4,4,3),cex=1.0)
plot(train.cal,lwd=2,lty=1,errbar.col=c(rgb(0,118,192,maxColorValue = 255)),
     xlim = c(0,1),ylim = c(0,1),xlab ="Model Predicted Survival(Train)",subtitles=FALSE,
     ylab="Actual Survival",col=c(rgb(255,0,0,maxColorValue =255)))
abline(0,1,lty=3,lwd=2,col=c(rgb(0,0,0,maxColorValue= 255)))


# 验证集校准曲线
val.cal <- calibrate(coxm, u=inc, method = "boot",cmethod = 'KM', m = 80, new.data=new_val, B=200)
par(mar=c(7,4,4,3),cex=1.0)
plot(val.cal,lwd=1,lty=1,errbar.col=c(rgb(0,0,0,maxColorValue = 255)),
     xlim = c(0,1),ylim = c(0,1),xlab ="Model Predicted Survival(Val)",,subtitles=FALSE,
     ylab="Actual Survival",col=c(rgb(255,0,0,maxColorValue =255)))
abline(0,1,lty=3,lwd=2,col=c(rgb(0,0,0,maxColorValue= 255)))



######################      关于 cox分析 的验证     ####################
pred_f_training<-predict(coxm,new_train,type="lp")#!!!type="lp",是他没错
print(pred_f_training)
data_table<-data.frame(time=new_train[,"os_time"],status=new_train[,"censor"],score=pred_f_training)

new_train$prognsis <- ifelse(pred_f_training>0,"poor", "good")
fit.surv <- survfit(Surv(os_time,censor==0)~prognsis, data=new_train)
ggsurvplot(fit.surv, conf.int=TRUE)


#这部分如果预测不是1年、3年、5年的话，需要调整times参数中的数值
time_roc_res <- timeROC(
  T = data_table$time,
  delta = data_table$status==0,
  marker = data_table$score,
  cause = 1,
  weighting="marginal",
  times = c(12, 3*12, 5*12),
  ROC = TRUE,
  iid = TRUE
)
time_ROC_df <- data.frame(
  TP_1year = time_roc_res$TP[, 1],
  FP_1year = time_roc_res$FP[, 1],
  TP_3year = time_roc_res$TP[, 2],
  FP_3year = time_roc_res$FP[, 2],
  TP_5year = time_roc_res$TP[, 3],
  FP_5year = time_roc_res$FP[, 3]
)

print(time_roc_res)

#绘制1、3、5年生存率预测的ROC图
ggplot(data = time_ROC_df, smooth=T) +
  geom_line(aes(x = FP_1year, y = TP_1year), size = 1, color = "#BC3C29FF") +
  geom_line(aes(x = FP_3year, y = TP_3year), size = 1, color = "#0072B5FF") +
  geom_line(aes(x = FP_5year, y = TP_5year), size = 1, color = "#E18727FF") +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  annotate("text",
           x = 0.75, y = 0.25, size = 4.5,
           label = paste0("AUC at 1 year = ", sprintf("%.3f", time_roc_res$AUC[[1]])), color = "#BC3C29FF"
  ) +
  annotate("text",
           x = 0.75, y = 0.15, size = 4.5,
           label = paste0("AUC at 3 years = ", sprintf("%.3f", time_roc_res$AUC[[2]])), color = "#0072B5FF"
  ) +
  annotate("text",
           x = 0.75, y = 0.05, size = 4.5,
           label = paste0("AUC at 5 years = ", sprintf("%.3f", time_roc_res$AUC[[3]])), color = "#E18727FF"
  ) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme(
    axis.text = element_text(face = "bold", size = 11, color = "black"),
    axis.title.x = element_text(face = "bold", size = 14, color = "black", margin = ggplot2::margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(face = "bold", size = 14, color = "black", margin = ggplot2::margin(c(0, 15, 0, 0)))
  )

###验证集
# new_bind <- rbind(new_val)


pred_f_val<-predict(coxm,new_val,type="lp")#!!!type="lp",是他没错
print(pred_f_val)
data_table<-data.frame(time=new_val[,"os_time"],status=new_val[,"censor"],score=pred_f_val)


#这部分如果预测不是1年、3年、5年的话，需要调整times参数中的数值
time_roc_res <- timeROC(
  T = data_table$time,
  delta = data_table$status==0,
  marker = data_table$score,
  cause = 1,
  weighting="marginal",
  times = c(12, 3*12, 5*12),
  ROC = TRUE,
  # iid = TRUE
)
time_ROC_df <- data.frame(
  TP_1year = time_roc_res$TP[, 1],
  FP_1year = time_roc_res$FP[, 1],
  TP_3year = time_roc_res$TP[, 2],
  FP_3year = time_roc_res$FP[, 2],
  TP_5year = time_roc_res$TP[, 3],
  FP_5year = time_roc_res$FP[, 3]
)

write.csv(time_ROC_df, file="C:/Users/12828/OneDrive/文档/骨肉瘤论文/生存率预测/ROC-结合影像和临床.csv",quote=F,row.names = F)

print(time_roc_res)

#绘制1、3、5年生存率预测的ROC图
ggplot(data = time_ROC_df) +
  geom_line(aes(x = FP_1year, y = TP_1year), size = 1, color = "#BC3C29FF") +
  geom_line(aes(x = FP_3year, y = TP_3year), size = 1, color = "#0072B5FF") +
  geom_line(aes(x = FP_5year, y = TP_5year), size = 1, color = "#E18727FF") +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  annotate("text",
           x = 0.75, y = 0.25, size = 4.5,
           label = paste0("AUC at 1 year = ", sprintf("%.3f", time_roc_res$AUC[[1]])), color = "#BC3C29FF"
  ) +
  annotate("text",
           x = 0.75, y = 0.15, size = 4.5,
           label = paste0("AUC at 3 years = ", sprintf("%.3f", time_roc_res$AUC[[2]])), color = "#0072B5FF"
  ) +
  annotate("text",
           x = 0.75, y = 0.05, size = 4.5,
           label = paste0("AUC at 5 years = ", sprintf("%.3f", time_roc_res$AUC[[3]])), color = "#E18727FF"
  ) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme(
    axis.text = element_text(face = "bold", size = 11, color = "black"),
    axis.title.x = element_text(face = "bold", size = 14, color = "black", margin = ggplot2::margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(face = "bold", size = 14, color = "black", margin = ggplot2::margin(c(0, 15, 0, 0)))
  )


## C-index曲线
c_index  <- cindex(list("Cox(12 variables)"=coxm),
                   formula=Surv(os_time,censor==0)~.,
                   data=new_val,
                   eval.times=seq(12,5*12,1.2))
par(mgp=c(3.1,0.8,0),mar=c(5,5,3,1),cex.axis=0.8,cex.main=0.8)
plot(c_index,xlim = c(0,65),legend.x=1,legend.y=1,legend.cex=0.8)








