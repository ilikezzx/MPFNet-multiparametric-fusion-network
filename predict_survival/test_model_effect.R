rm(list=ls()) # 清空数据
library(car)
library(rms)   ###加载rms包#
library(pROC)
library(rmda)

data_dir <- choose.dir(default = "C:\\osteosarcoma\\os_survival", caption = "选择数据存放的文件夹目录")
output_dir <- choose.dir(default = "C:\\osteosarcoma\\os_survival", caption = "选择导出图片的文件夹目录")

# 读取训练集和测试集
training_dataset_path <- choose.files(default = data_dir, caption = "请选训练集数据文件的csv文件。",
                                      multi = TRUE, filters = Filters,
                                      index = nrow(Filters))

#读取数据
training_dataset<- read.csv(training_dataset_path, header = TRUE,sep=",", stringsAsFactors=FALSE)
#查看数据
print(paste0("该训练集有 ",dim(training_dataset)[1]," 个样本；",dim(training_dataset)[2]," 个变量"))

#选择文件路径
validation_dataset_path <- choose.files(default = data_dir, caption = "请选训练集数据文件的txt文件。",
                                        multi = TRUE, filters = Filters,
                                        index = nrow(Filters))
#读取数据
validation_dataset<- read.csv(validation_dataset_path, header = TRUE,sep=",", stringsAsFactors=FALSE)
#查看数据
print(paste0("该验证集有 ",dim(validation_dataset)[1]," 个样本；",dim(validation_dataset)[2]," 个变量"))

# 设定环境参数
ddist <- datadist(training_dataset)
options(datadist='ddist')


f <- lrm(five_years_survival~ sex+post_chemotherapy+lung_metastases+Logistic_Score,data=training_dataset, x=TRUE, y=TRUE,maxit=1000)   
summary(f)   # 也能用此函数看具体模型情况，模型的系数，置信区间等
print(f)

#nomogram计算部分，此处的f_lrm及对应的多因素logistic回归函数。
pdf(file=paste("./nomogram.pdf", sep = ""),width=10,height=5) 
nomogram <- nomogram(f,fun=function(x)1/(1+exp(-x)), ##逻辑回归计算公式
                     fun.at = c(0.01,0.1,0.3,0.5,0.8,0.9,0.99),#风险轴刻度
                     funlabel = "Prob of death", #风险轴便签
                     lp=F,  ##是否显示系数轴
                     conf.int = F, ##每个得分的置信度区间，用横线表示,横线越长置信度越
                     abbrev = F#是否用简称代表因子变量
)
#绘制nomogram
plot(nomogram)
dev.off()


##训练集中的ROC
pred_f_training<-predict(f,training_dataset)

#下方参数中Death需改为你的研究的结局变量名
modelroc <- roc(training_dataset$five_years_survival,pred_f_training)
#绘制ROC
pdf(file=paste(output_dir, "\\ROC_training.pdf", sep = ""),width=10,height=10) 
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     print.thres=TRUE)
dev.off()

##验证集中的ROC
pred_f_validation<-predict(f,validation_dataset)
#下方参数中Death需改为你的研究的结局变量名
modelroc <- roc(validation_dataset$five_years_survival,pred_f_validation)
#绘制ROC
pdf(file=paste(output_dir, "\\ROC_testing.pdf", sep = ""),width=10,height=10) 
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     print.thres=TRUE)
dev.off()

# 训练集的校准曲线 CalibrateCurve
cal <- calibrate(f)
pdf(file=paste(output_dir, "\\calibrate_training.pdf", sep = ""),width=10,height=10) 
plot(cal)
dev.off()

# 测试集的校准曲线
fit.vad<-lrm(validation_dataset$five_years_survival~pred_f_validation,data=validation_dataset,x=T,y=T)
pdf(file=paste(output_dir, "\\calibrate_testing.pdf", sep = ""),width=10,height=10) 
cal <- calibrate(fit.vad)
plot(cal)
dev.off()

##训练集决策曲线DCA
pdf(file=paste(output_dir, "\\DCA_training.pdf", sep = ""),width=10,height=10) 
DCA_training<- decision_curve(five_years_survival ~ sex+post_chemotherapy+lung_metastases+Logistic_Score
                              ,data = training_dataset
                              #,policy = "opt-in"
                              ,study.design = 'cohort')
plot_decision_curve(DCA_training,curve.names= c('Nomogram model'))
dev.off()

#验证集决策曲线DCA
pdf(file=paste(output_dir, "\\DCA_testing.pdf", sep = ""),width=10,height=10) 
DCA_training<- decision_curve(five_years_survival ~ sex+post_chemotherapy+lung_metastases+Logistic_Score
                              ,data = validation_dataset
                              #,policy = "opt-in"
                              ,study.design = 'cohort')
plot_decision_curve(DCA_training,curve.names= c('Nomogram model'))
dev.off()
