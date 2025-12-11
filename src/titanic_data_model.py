import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
train_filename='train.csv'
test_filename='test.csv'
train_data=pd.read_csv(train_filename)
test_data=pd.read_csv(test_filename)
train_data.info()#观察数据特征

#可以看到存在车票相同、费用相同，cabin号多个的乘客，因此我们理解为，这是包括多个座位的车票
#为了保持维度一致，创建一列车票价格，用于后续分析车票平均价等等信息。
#对于购买多张票的人计算票价，创建新列Fare_mean保存所有票价
train_data['Ticket_count']=train_data['Ticket'].map(train_data['Ticket'].value_counts()) 
train_data['Fare_mean']=train_data['Fare']/train_data['Ticket_count']

#太复杂了，换思路，先给age分组，然后填充空age,用平均值
bins=[0,6,14,18,60,120]
labels=['baby','child','teen','aldult','old']
train_data['age_group']=pd.cut(train_data['Age'],bins=bins,labels=labels,right=False)
train_data
train_data.loc[train_data['Age'].isna(),'Age']=train_data['Age'].mean()
train_data.loc[train_data['age_group'].isna(),'age_group']='aldult'

#观察车票后缀是数字，说不定有规律，所以拆分
#处理车票号码
mask=~train_data['Ticket'].str.isdigit()#不是纯数字的车票号码拆分成两列
split_result = train_data.loc[mask, 'Ticket'].str.split(' ', n=1, expand=True)

train_data.loc[mask, 'ticket_pre'] = split_result[0]# 赋值到新列
train_data.loc[mask, 'ticket_last'] = split_result[1]

pure_digit_mask = train_data['Ticket'].str.isdigit()# 对于纯数字的票号，前缀为空，后缀为票号本身
train_data.loc[pure_digit_mask, 'ticket_last'] = train_data.loc[pure_digit_mask, 'Ticket']
train_data.loc[pure_digit_mask, 'ticket_pre'] = ''
#有的车票号码拆分后，后项还有字母，因为数量少，直接去除票号码中的字母
train_data['ticket_last'] = train_data['ticket_last'].astype(str)
train_data['ticket_last'] = train_data['ticket_last'].str.replace(r'[a-zA-Z\s]', '', regex=True)
train_data['ticket_last']=pd.to_numeric(train_data['ticket_last'],errors='coerce', downcast='integer')
#ticket尾号有空值，处理空值为中位数
median_value=train_data['ticket_last'].median()
train_data.loc[train_data['ticket_last'].isna(),'ticket_last']=median_value

#同理拆分车厢号备用
train_data['Cabin_num']=train_data['Cabin'].str[::-1].str.extract(r'([a-zA-Z])', expand=False)
train_data['Cabin_num']

train_data['Cabin'] = train_data['Cabin'].astype(str)
train_data['Cabin'] = train_data['Cabin'].str.replace(r'[a-zA-Z]', '', regex=True)

#拆分姓名和称呼
split_result = train_data['Name'].str.split(',', n=1, expand=True)
train_data['f_name']=split_result[0]
train_data['t_name']=split_result[1]
split_result1=train_data['t_name'].str.split('.', n=1, expand=True)
train_data['t_name']=split_result1[0]
train_data['l_name']=split_result1[1]


train_data#观察数据，查看可能存在的规律，探索性数据分析
train_data.info()
#cabin缺失值太多，看看有没有规律能够填充上去的
train_data[(train_data['Cabin'].notna()) &(train_data['Fare']!=0)].groupby(['Cabin_num','Pclass']).agg({
    'Fare_mean':['min','mean','max','count']
})

train_data[(train_data['Fare']!=0)].groupby(['Pclass']).agg({
    'Fare_mean':['min','mean','max','count']
})

train_data[(train_data['Cabin'].notna()) &(train_data['Fare']!=0)].groupby(['Pclass','Cabin_num']).agg({
    'Fare_mean':['min','mean','max','count'],
    'Cabin_num':['count']
})
#可以观察到只有CDE包含了三种等级的舱位，其他的只有一种舱位。但不知道怎么用
train_data['Cabin_Pclass'] = train_data['Cabin_num'].astype(str) + train_data['Pclass'].astype(str)
train_data[train_data['Cabin_num'].notna()&(train_data['Fare_mean']!=0)].groupby('Cabin_Pclass').agg({
    'Fare_mean':['mean','min','max','count'],
    'Survived':['sum']
})#可以观察到，不同的票价对应不同的舱位，不同的舱位存活率不同。再画一个散点图，观察舱位和票价之间的关系，说不定能推测出空白的舱位号，并且可以把类似票价的舱位合并

mask=train_data['Cabin_num'].notna()
plt.scatter(train_data.loc[mask,'Cabin_Pclass'], train_data.loc[mask,'Fare_mean'])#观察图可得：A1B1C1D1E1是高票价舱大于25.57，其他为第票价舱。由此来给数据预测舱位
high=['A1','B1','C1','D1','E1']
for i in train_data['Cabin_Pclass']:
    if len(i)>0:
        if i in high:
            train_data['fare_class']='high'
        else:
            train_data['fare_class']='low'
    elif train_data['Fare_mean']>25.57:
        train_data['fare_class']='high'
    else:
            train_data['fare_class']='low'

#训练准备
#创建pclass、tname、embarked\age_group\性别的独热编码备用
pclass_mapping=pd.get_dummies(train_data['Pclass'])
pclass_mapping

tname_mapping=pd.get_dummies(train_data['t_name'])
tname_mapping

Embarked_mapping=pd.get_dummies(train_data['Embarked'])
Embarked_mapping

agegroup_mapping=pd.get_dummies(train_data['age_group'])
agegroup_mapping

sex_mapping=pd.get_dummies(train_data['Sex'])
sex_mapping

fareclass_mapping=pd.get_dummies(train_data['fare_class'])
fareclass_mapping


#做一个训练数据，组合独热编码查看效果
new_traindata_X=train_data[['SibSp','Parch','ticket_last','Fare_mean']]#数字类型的列


#建立六种不同维度的训练数据
new_traindata_X1=np.hstack((new_traindata_X,pclass_mapping))
new_traindata_X2=np.hstack((new_traindata_X1,tname_mapping))
new_traindata_X3=np.hstack((new_traindata_X2,Embarked_mapping))
new_traindata_X4=np.hstack((new_traindata_X3,agegroup_mapping))
new_traindata_X5=np.hstack((new_traindata_X4,sex_mapping))
new_traindata_X6=np.hstack((new_traindata_X5,fareclass_mapping))

new_traindata_Y=train_data['Survived']
#维数约减
#用协方差矩阵判断能否进行维数约减
cov_data_x6=np.corrcoef(new_traindata_X6.T)
cov_data_x6

#画一个矩阵热力图显示协方差矩阵
plt.figure(figsize=(100, 100)) 
img=plt.matshow(cov_data_x6,cmap=plt.cm.rainbow)
fig = plt.gcf()
fig.set_size_inches(40, 32)  #修改画布大小
plt.colorbar(img,ticks=[-1,0,1],fraction=0.045)# 为热图添加颜色条（图例
for x in range(cov_data_x6.shape[0]):
    for y in range(cov_data_x6.shape[1]):
        plt.text(x,y,"%.2f"% cov_data_x6[x,y],size=8,color='black',ha='center',va='center')
plt.show()
#通过矩阵热图观察大概只有11维的关联性比较大，用PCA分析主成分
from sklearn.decomposition import PCA 
pca_x6_11=PCA(n_components=11)
x6_11w=pca_x6_11.fit_transform(new_traindata_X6)
x6_11w.shape
pca_x6_11.explained_variance_ratio_.sum()#结果是：0.9999999999993804 非常好

#2维
pca_x6_2=PCA(n_components=2)
x6_2w=pca_x6_2.fit_transform(new_traindata_X6)
x6_2w.shape
pca_x6_2.explained_variance_ratio_.sum()
#查看散点图
plt.scatter(x6_2w[:,0],x6_2w[:,1],c=new_traindata_Y,alpha=0.8,s=60,marker='o',edgecolors='white')#分类结果并不理想，改用有监督的降维
plt.show()

#用LDA试试
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_x6_2w=LinearDiscriminantAnalysis(n_components=1)
x6_2w_lda=lda_x6_2w.fit_transform(new_traindata_X6,new_traindata_Y)
plt.scatter(x6_2w_lda,new_traindata_Y,c=new_traindata_Y,alpha=0.8,edgecolors='None')
plt.hist(x6_2w_lda[new_traindata_Y==0], bins=30, alpha=0.5, label='类别0')
plt.hist(x6_2w_lda[new_traindata_Y==1], bins=30, alpha=0.5, label='类别1')
plt.xlabel('LDA一维投影值')
plt.ylabel('样本数量')
plt.title('类别在LDA一维投影上的分布')
plt.legend()
plt.grid(True)
plt.show()#效果好一些了但是也一般，减少一些无关特征试试

lda_x_1w=LinearDiscriminantAnalysis(n_components=1)
x_1w_lda=lda_x_1w.fit_transform(new_traindata_X,new_traindata_Y)
plt.scatter(x_1w_lda,new_traindata_Y,c=new_traindata_Y,alpha=0.8,edgecolors='None')
plt.hist(x_1w_lda[new_traindata_Y==0], bins=30, alpha=0.5, label='类别0')
plt.hist(x_1w_lda[new_traindata_Y==1], bins=30, alpha=0.5, label='类别1')
plt.xlabel('LDA一维投影值')
plt.ylabel('样本数量')
plt.title('类别在LDA一维投影上的分布')
plt.legend()
plt.grid(True)
plt.show()#效果更差

lda_x3_1w=LinearDiscriminantAnalysis(n_components=1)
x3_1w_lda=lda_x3_1w.fit_transform(new_traindata_X3,new_traindata_Y)
plt.scatter(x3_1w_lda,new_traindata_Y,c=new_traindata_Y,alpha=0.8,edgecolors='None')
plt.hist(x3_1w_lda[new_traindata_Y==0], bins=30, alpha=0.5, label='类别0')
plt.hist(x3_1w_lda[new_traindata_Y==1], bins=30, alpha=0.5, label='类别1')
plt.xlabel('LDA一维投影值')
plt.ylabel('样本数量')
plt.title('类别在LDA一维投影上的分布')
plt.legend()
plt.grid(True)
plt.show()#和X6差不多

lda_x4_1w=LinearDiscriminantAnalysis(n_components=1)
x4_1w_lda=lda_x4_1w.fit_transform(new_traindata_X4,new_traindata_Y)
plt.scatter(x4_1w_lda,new_traindata_Y,c=new_traindata_Y,alpha=0.8,edgecolors='None')
plt.hist(x4_1w_lda[new_traindata_Y==0], bins=30, alpha=0.5, label='类别0')
plt.hist(x4_1w_lda[new_traindata_Y==1], bins=30, alpha=0.5, label='类别1')
plt.xlabel('LDA一维投影值')
plt.ylabel('样本数量')
plt.title('类别在LDA一维投影上的分布')
plt.legend()
plt.grid(True)
plt.show()#对比后发现pclass_mapping这个字段作用不大

#去掉这个组再逐个试试效果
new_traindata_X2=np.hstack((new_traindata_X,tname_mapping))
new_traindata_X3=np.hstack((new_traindata_X2,Embarked_mapping))
new_traindata_X4=np.hstack((new_traindata_X3,agegroup_mapping))
new_traindata_X5=np.hstack((new_traindata_X4,sex_mapping))
new_traindata_X6=np.hstack((new_traindata_X5,fareclass_mapping))#6相对于5变化不明显，淘汰

new_traindata_X1=np.hstack((new_traindata_X,sex_mapping))
new_traindata_X2=np.hstack((new_traindata_X1,tname_mapping))
new_traindata_X3=np.hstack((new_traindata_X2,Embarked_mapping))#这两个数据集效果还行，算是最佳了，大致分开但中间部分有重叠
new_traindata_X4=np.hstack((new_traindata_X3,agegroup_mapping))#这两个数据集效果还行，算是最佳了，大致分开但中间部分有重叠 

new_traindata_X7=np.delete(new_traindata_X4,2,axis=1)
new_traindata_X8=np.delete(new_traindata_X7,2,axis=1)
new_traindata_X8=np.delete(new_traindata_X8,1,axis=1)#去掉了ticketlast和平均票价和父母列效果还略好一点，目前的最佳
new_traindata_X8_noage=new_traindata_X8[:,:-5]#去掉了年龄分组和平均票价和父母列效果还略好一点，目前的最佳
new_traindata_X10=np.delete(new_traindata_X2,2,axis=1)#加入模型训练后最好的结果

lda_x10_1w=LinearDiscriminantAnalysis(n_components=1)
x10_1w_lda=lda_x10_1w.fit_transform(new_traindata_X10,new_traindata_Y)
plt.scatter(x10_1w_lda,new_traindata_Y,c=new_traindata_Y,alpha=0.8,edgecolors='None')
plt.hist(x10_1w_lda[new_traindata_Y==0], bins=30, alpha=0.5, label='类别0')
plt.hist(x10_1w_lda[new_traindata_Y==1], bins=30, alpha=0.5, label='类别1')
plt.xlabel('LDA一维投影值')
plt.ylabel('样本数量')
plt.title('类别在LDA一维投影上的分布')
plt.legend()
plt.grid(True)
plt.show()

#用x8来训练一下试试：
from sklearn.model_selection import train_test_split

# 用LDA之前的原始特征划分
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    new_traindata_X8,  # 原始特征，改成x10后是最佳的，X2也差不多
    new_traindata_Y,   # 目标变量
    test_size=0.2,     # 20%测试集
    random_state=42,
    stratify=new_traindata_Y  # 保持类别比例
)

# 在训练集上训练LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_orig, y_train)

# 用训练好的LDA转换测试集
X_test_lda = lda.transform(X_test_orig)

from sklearn.linear_model import LogisticRegression#经过对比，用x2和x10做线性回归后，模型的效果最好
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 训练逻辑回归
lr_model = LogisticRegression(random_state=42)#用未经过lda处理的数据训练效果不如处理后的，说明还是有用
lr_model.fit(X_train_lda, y_train)

# 预测
y_train_pred = lr_model.predict(X_train_lda)
y_test_pred = lr_model.predict(X_test_lda)

print(f"训练集准确率: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_test_pred):.4f}")
