###############################################################
#   这个写关于特征筛选 + 回归分析
#    两种特征选择方式，随机森林(根据重要性排序)+Lasso去零的特征
###############################################################

# 下载安装包
#install.packages('glmnet')
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

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

######################      读取数据     ####################
# 读取训练集
training_dataset_path <- choose.files(caption = "请选训练集数据文件的csv文件。",
                                      multi = TRUE, filters = Filters,
                                      index = nrow(Filters))
#读取数据
training_dataset<- read.csv(training_dataset_path, header = TRUE,sep=",", stringsAsFactors=FALSE)
train_img_features <- data.matrix(training_dataset[, 1:851])
train_clin_features <- data.matrix(training_dataset[, 852:856])
train_label <- data.matrix(training_dataset[, 857:857])
colnames(train_label)[1] <- 'five_os'
#查看数据
print(paste0("该训练集有 ",dim(training_dataset)[1]," 个样本；",dim(training_dataset)[2]," 个变量"))


# 测试集
validation_dataset_path <- choose.files(caption = "请选训练集数据文件的txt文件。",
                                        multi = TRUE, filters = Filters,
                                        index = nrow(Filters))
# 读取数据
validation_dataset<- read.csv(validation_dataset_path, header = TRUE,sep=",", stringsAsFactors=FALSE)
val_img_features <- data.matrix(validation_dataset[, 1:851])
val_clin_features <- data.matrix(validation_dataset[, 852:856])
val_label <- data.matrix(validation_dataset[, 857:857])
colnames(val_label)[1] <- 'five_os'
# 查看数据
print(paste0("该验证集有 ",dim(validation_dataset)[1]," 个样本；",dim(validation_dataset)[2]," 个变量"))


######################      Lasso  特征选择     ####################
train_fit <- glmnet(train_img_features, train_label, alpha=1,family = 'binomial')
#我们也可以绘图展示根据lambda变化情况每一个特征的系数变化
plot(train_fit, xvar = "lambda")
# 使用area under the ROC curve, CV 选择压缩参数lambda
# 再设置一次set.seed
set.seed(1)
train_fit_cv <- cv.glmnet(train_img_features, train_label, alpha=1, family = 'binomial', type.measure='auc', nfolds=4)
plot(train_fit_cv)
coef(train_fit_cv, s = "lambda.min")


# 纯影像学训练集的效果判断
train_pred <- data.matrix(sigmoid(predict(train_fit_cv, train_img_features)))
colnames(train_pred)[1] <- 'imaging_score'
# 训练集 ROC
train_modelroc <- roc(train_label, sigmoid(train_pred))
plot(train_modelroc, print.auc=TRUE, auc.polygon=TRUE)



# 纯影像学测试集的效果判断
val_pred <- data.matrix(predict(train_fit_cv, val_img_features))
colnames(val_pred)[1] <- 'imaging_score'
# 测试集 ROC
val_modelroc <- roc(val_label,sigmoid(val_pred))
plot(val_modelroc, print.auc=TRUE, auc.polygon=TRUE)



# # 保存非零相关性自变量列索引
# dd = data.matrix(coef(train_fit_cv, s = "lambda.min"))
# non_zero_index <- c()
# for(i in 2:nrow(dd)){
#   if(dd[i] > 0){
#     non_zero_index <- append(non_zero_index, i-1)
#   }
# }
# 
# nonzero_train_features <- data.frame(train_img_features[, non_zero_index])
# nonzero_val_features <- data.frame(val_img_features[, non_zero_index])

######################      回归分析     ####################
now_train_dataset <- data.frame(cbind(train_pred, train_clin_features, train_label))
now_val_dataset <- data.frame(cbind(val_pred, val_clin_features, val_label))

# 数据集操作，将离散型变量改成哑变量
now_train_dataset$sex <- factor(now_train_dataset$sex,levels = c(1,0),labels = c("男", "女"))
now_train_dataset$lung_metastases <- factor(now_train_dataset$lung_metastases,levels = c(1,0),labels = c("已肺转移", "未肺转移"))
now_val_dataset$sex <- factor(now_val_dataset$sex,levels = c(1,0),labels = c("男", "女"))
now_val_dataset$lung_metastases <- factor(now_val_dataset$lung_metastases,levels = c(1,0),labels = c("已肺转移", "未肺转移"))


####      多因素Logistic回归     ######
ddist <- datadist(now_train_dataset)
options(datadist='ddist')

f <- lrm(five_os~imaging_score+sex+lung_metastases,data=now_train_dataset, x=TRUE, y=TRUE,maxit=1000)
summary(f)   # 也能用此函数看具体模型情况，模型的系数，置信区间等
print(f)

vif(f)
f2 <-step(f)

#绘制nomogram
nomogram <- nomogram(f,fun=function(x)1/(1+exp(-x)), ##逻辑回归计算公式
                     fun.at = c(0.1,0.6,0.99),#风险轴刻度
                     funlabel = "Prob of survival", #风险轴便签
                     lp=F,  ##是否显示系数轴
                     conf.int = F, ##每个得分的置信度区间，用横线表示,横线越长置信度越
                     abbrev = F#是否用简称代表因子变量
)
plot(nomogram)


##训练集中的ROC
pred_f_training<-predict(f,now_train_dataset)
modelroc <- roc(now_train_dataset$five_os,pred_f_training)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2), print.thres=TRUE)

# 验证集中的ROC
pred_f_validation<-predict(f,now_val_dataset)
modelroc <- roc(now_val_dataset$five_os,pred_f_validation)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     print.thres=TRUE)

# 训练集的校准曲线 CalibrateCurve
cal <- calibrate(f)
plot(cal)

# 测试集的校准曲线
fit.vad<-lrm(now_val_dataset$five_os~pred_f_validation,data=now_val_dataset, x=TRUE, y=TRUE)
cal <- calibrate(fit.vad)
plot(cal)

##训练集决策曲线DCA
DCA_training<- decision_curve(five_os ~ imaging_score+age+sex+lung_metastases
                              ,data = now_train_dataset
                              #,policy = "opt-in"
                              ,study.design = 'cohort')
plot_decision_curve(DCA_training,curve.names= c('Nomogram model'))

#验证集决策曲线DCA
DCA_training<- decision_curve(five_os ~ imaging_score+age+sex+lung_metastases
                              ,data = now_val_dataset
                              #,policy = "opt-in"
                              ,study.design = 'cohort')
plot_decision_curve(DCA_training,curve.names= c('Nomogram model'))


####      随机森林     ######
gx.rf<-randomForest(five_os~imaging_score+sex+age+volume+lung_metastases,data=now_train_dataset,importance=TRUE, ntree=1000)
print(gx.rf)
importance(gx.rf)
varImpPlot(gx.rf)


#使用训练集，查看预测精度
train_predict <- data.matrix(predict(gx.rf, now_train_dataset))
plot(now_train_dataset$five_os, train_predict, main = '训练集', xlab = 'true five_os', ylab = 'Predict')
abline(0, 1)
train_modelroc <- roc(train_label, round(train_predict))
plot(train_modelroc, print.auc=TRUE, auc.polygon=TRUE)


#使用测试集，评估预测性能
val_predict <- predict(gx.rf, now_val_dataset)
plot(now_val_dataset$five_os, val_predict, main = '测试集', xlab = 'true five_os', ylab = 'Predict')
abline(0, 1)
val_modelroc <- roc(val_label, round(val_predict))
plot(val_modelroc, print.auc=TRUE, auc.polygon=TRUE)



