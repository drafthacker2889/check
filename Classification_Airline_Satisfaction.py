#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, lars_path
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve,f1_score, roc_auc_score, roc_curve, log_loss,classification_report

from ipywidgets import interactive

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import warnings
warnings.filterwarnings("ignore")



# In[2]:


df = pd.read_csv('Airline_Dataset.csv')


# # Data Cleaning
# #### The columns in the dataframe are cleaned and reorganized:
# - Column names are renamed.
# - Elements in Features 'Customer Type' and 'Class' are renamed.
# - Rows with Null values are removed.
# - Rows with scores of 0 in the survey of satisfaction are removed (Customers probably did not indicate).
# - Departure Delay and Arrival Delay are combined.
# - Satisfaction target is relabelled as 0 and 1.

# In[3]:


df


# In[4]:


df.info()


# In[5]:


df['Customer Type'] = df['Customer Type'].map({'Loyal Customer':'Returning Customer','disloyal Customer':'First-time Customer'})


# In[6]:


df = df.dropna(axis=0)


# In[7]:


df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].astype('float')


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df = df.rename(columns={'Leg room service':'Leg room'})


# In[11]:


from string import capwords
df.columns = [capwords(i) for i in df.columns]
df = df.rename(columns={'Departure/arrival Time Convenient':'Departure/Arrival Time Convenience'})


# In[12]:


df


# In[13]:


df = df[(df['Inflight Wifi Service']!=0)&(df['Departure/Arrival Time Convenience']!=0)&(df['Ease Of Online Booking']!=0)&(df['Gate Location'])&(df['Food And Drink']!=0)&(df['Online Boarding']!=0)&(df['Seat Comfort']!=0)&(df['Inflight Entertainment']!=0)&(df['On-board Service']!=0)&(df['Leg Room']!=0)&(df['Baggage Handling']!=0)&(df['Checkin Service']!=0)&(df['Inflight Service']!=0)&(df['Cleanliness']!=0)]


# In[14]:


df['Satisfaction'] = df['Satisfaction'].map({'satisfied':1,'neutral or dissatisfied':0})
df = df.reset_index()
df = df.drop('index',axis=1)
df['Total Delay'] = df['Departure Delay In Minutes'] + df['Arrival Delay In Minutes']


# In[15]:


DF = df.copy()
df = df.drop('Id',axis=1)


# In[16]:


df = df.reindex(columns=['Satisfaction']+list(df.columns)[:-2]+['Total Delay'])
df = df.drop(['Departure Delay In Minutes','Arrival Delay In Minutes'],axis=1)


# In[17]:


df['Satisfaction'].value_counts(normalize=True)


# In[18]:


df['Class'] = df['Class'].map({'Eco':'Economy','Eco Plus':'Economy','Business':'Business'})


# In[19]:


df


# # Exploratory Data Analysis and Feature Selection
# 
# #### Create visualizations to first understand business problem, and also identify important features for model building:
# - Find out proportion of classes in target, and split them by Type of Travel and Type of Customers (To understand trend of satisfaction - useful later in model evaluation)
# - Identify feature significance for model through visualizing KDE plots, LASSO path and heatmap.
# - After evaluation and discreet selection, I have decided to drop 'Gender, 'Total Delay','Flight Distance','Age','Gate Location' and 'Departure/Arrival Time Convenience'

# In[20]:


sns.set(style='white',font_scale=1.1)
fig = plt.figure(figsize=[5,6])
ax = sns.countplot(data=df,x='Satisfaction',palette='coolwarm')
ax.set_xticklabels(['Neutral/Dissatisfied','Satisfied'])
for p in ax.patches:
        ax.annotate(str(p.get_height())+' ('+str((p.get_height()/len(df)*100).round(1))+'%)', (p.get_x()+0.1, p.get_height()+400))
plt.xlabel('Satisfaction',weight='bold',fontsize='15')   
plt.ylabel('No. of Passengers',weight='bold',fontsize='15')   
sns.despine()
plt.savefig('targetplot1.png',transparent=True, bbox_inches='tight')


# In[21]:


sns.set(style='white',font_scale=1.1)
fig = plt.figure(figsize=[5,6])
ax = sns.countplot(data=df,x='Satisfaction',hue='Type Of Travel',palette='Blues')
ax.set_xticklabels(['Neutral/Dissatisfied','Satisfied'])
for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.08, p.get_height()+200))
plt.xlabel('Satisfaction',weight='bold',fontsize='15')   
plt.ylabel('No. of Passengers',weight='bold',fontsize='15')   
sns.despine()
plt.savefig('targetplot2.png',transparent=True, bbox_inches='tight')


# In[22]:


sns.set(style='white',font_scale=1.1)
fig = plt.figure(figsize=[5,6])
ax = sns.countplot(data=df,x='Satisfaction',hue='Customer Type',palette='Greens')
ax.set_xticklabels(['Neutral/Dissatisfied','Satisfied'])
for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.08, p.get_height()+200))
plt.xlabel('Satisfaction',weight='bold',fontsize='15')   
plt.ylabel('No. of Passengers',weight='bold',fontsize='15')   
plt.legend(loc="upper right", bbox_to_anchor=(1.6, 0.2),fontsize=13)
sns.despine()
plt.savefig('targetplot3.png',transparent=True, bbox_inches='tight')


# In[23]:


df1 = pd.get_dummies(df,columns=['Gender','Customer Type','Type Of Travel','Class'],drop_first=True)
df1


# In[24]:


df['Inflight Wifi Service'].value_counts()


# In[25]:


group = df1.groupby(['Satisfaction','Class_Economy'])['Class_Economy'].count()
group


# In[98]:


sns.set(style='white',font_scale=1.5)
fig = plt.figure(figsize=[30,20])
for i in range(20):
    fig.add_subplot(4, 5, i+1)
    sns.kdeplot(data=df1,x=df1.columns[i+1],hue='Satisfaction')
    if i == 16:
        plt.xlim([-50,300])
    sns.despine()
    plt.savefig('kdeplot.png',transparent=True, bbox_inches='tight')


# In[27]:


df1 = df1.drop('Gender_Male',axis=1)


# In[28]:


sns.set(style='white',font_scale=1.5)
fig = plt.figure(figsize=[30,20])
for i in range(20):
    fig.add_subplot(4, 5, i+1)
    sns.kdeplot(data=df1,x=df1.columns[i+1],hue='Satisfaction')
    if i == 16:
        plt.xlim([-50,300])
    sns.despine()


# In[29]:


corr_matrix = df1.corr()
corr_matrix


# In[30]:


sns.set(style='white',font_scale=2.2)
fig = plt.figure(figsize=[35,30])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(150, 0, as_cmap=True)
sns.heatmap(corr_matrix,cmap='seismic',linewidth=3,linecolor='white',vmax = 1, vmin=-1,mask=mask, annot=True,fmt='0.2f')
plt.title('Correlation Heatmap', weight='bold',fontsize=50)
plt.savefig('heatmap.png',transparent=True, bbox_inches='tight')


# In[31]:


y = df1['Satisfaction']
X = df1.drop('Satisfaction',axis=1)


# In[32]:


std = StandardScaler()
std.fit(X.values)
X_tr = std.transform(X.values)


# In[33]:


lasso_model = Lasso(alpha = 0.01)    
selected_columns = list(X.columns)
lasso_model.fit(X, y)
list(zip(selected_columns, lasso_model.coef_))


# In[34]:


alphas, _, coefs = lars_path(X_tr, y.values, method='lasso')

from cycler import cycler

# plotting the LARS path
sns.set(style='white',font_scale=2)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.rc('axes', prop_cycle =(cycler(color =['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A',
                                            '#D62728', '#FF9896', '#9467BD', '#C5B0D5', '#8C564B', '#C49C94',
                                            '#E377C2', '#F7B6D2', '#7F7F7F', '#C7C7C7', '#BCBD22', '#DBDB8D',
                                            '#17BECF', '#9EDAE5'])))

plt.figure(figsize=(15,10))
plt.plot(xx, coefs.T,linewidth=3.5)

ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.legend(X.columns,loc="upper right", bbox_to_anchor=(1.32, 0.9),fontsize=14)
sns.despine()
plt.savefig('lassoplot.png',transparent=True, bbox_inches='tight')


# In[35]:


## DROP ##
#Total Delay
#Flight Distance
#Age
#Gate Location
#df1 = df1.drop(['Total Delay','Flight Distance','Age','Gate Location'],axis=1)
#df1 = df1.drop(['Ease Of Online Booking','Food And Drink','Gate Location','Seat Comfort'],axis=1)
#df1 = df1.drop(['Total Delay','Age','Gate Location','Departure/Arrival Time Convenience'],axis=1)
df1 = df1.drop(['Total Delay','Flight Distance','Age','Gate Location','Departure/Arrival Time Convenience'],axis=1)


# # Model Selection
# #### Find out the best model for the data through Regularization, Cross Validation with evaluation with f1 score:
# - Logistic Regression (find out the best C)
# - KNN (find out the best k)
# - Gaussian Naive Bayes
# - Decision Trees (find out the best depth)
# - Random Forest (find out the best depth; no. of trees did not improve the model significantly)
# - Ensemble (Taking all the models with the best hyperparameters)
# 
# #### Random Forest is selected as the best model, but Simple Validation is conducted to tune probability threshold:
# - When threshold increased from 0.5 to 0.7, a better precision is obtained from 97% to 99%
# - We need the best precision for our business solution

# In[36]:


y = df1['Satisfaction']
X = df1.drop('Satisfaction',axis=1)


# In[37]:


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state = 71)


# In[38]:


Cvec = np.linspace(0.001,2,100)
logit_model = LogisticRegressionCV(Cs = Cvec, cv=kf,max_iter=10000)
logit_model.fit(X_train_val, y_train_val)
logit_model.C_


# In[39]:


logit_model = LogisticRegression(C=logit_model.C_[0],max_iter=10000)
Mean_AUC_Logit_CV = np.mean(cross_val_score(logit_model, X_train_val, y_train_val, cv=kf, scoring='roc_auc'))
Mean_AUC_Logit_CV 


# In[40]:


Mean_Precision_Logit_CV = np.mean(cross_val_score(logit_model, X_train_val, y_train_val, cv=kf, scoring='precision'))
Mean_Precision_Logit_CV 


# In[41]:


Mean_Recall_Logit_CV = np.mean(cross_val_score(logit_model, X_train_val, y_train_val, cv=kf, scoring='recall'))
Mean_Recall_Logit_CV 


# In[42]:


X_train_val = X_train_val.reset_index().drop('index',axis=1)
y_train_val = y_train_val.reset_index().drop('index',axis=1)


# In[43]:



#score = []
#for neighbors in range(5,11):
#    f1 = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=neighbors), X_train_val, y_train_val, cv=kf, scoring='f1'))
#    score.append(f1)
#    print(neighbors)
#best_neighbors = list(range(5,11))[np.argmax(score)]  
#best_f1 = max(score)
#print('The best k neighbours is {0} with f1-score of {1}'.format(best_neighbors,best_f1))   ''''''


# In[44]:


Mean_AUC_KNN_CV = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train_val, y_train_val, cv=kf, scoring='roc_auc'))
Mean_AUC_KNN_CV 


# In[45]:


Mean_Precision_KNN_CV = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train_val, y_train_val, cv=kf, scoring='precision'))
Mean_Precision_KNN_CV 


# In[46]:


Mean_Recall_KNN_CV = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train_val, y_train_val, cv=kf, scoring='recall'))
Mean_Recall_KNN_CV 


# In[47]:


Mean_AUC_NB_CV = np.mean(cross_val_score(GaussianNB(), X_train_val, y_train_val, cv=kf, scoring='roc_auc'))
Mean_AUC_NB_CV 


# In[48]:


Mean_Precision_NB_CV = np.mean(cross_val_score(GaussianNB(), X_train_val, y_train_val, cv=kf, scoring='precision'))
Mean_Precision_NB_CV 


# In[49]:


Mean_Recall_NB_CV = np.mean(cross_val_score(GaussianNB(), X_train_val, y_train_val, cv=kf, scoring='recall'))
Mean_Recall_NB_CV 


# In[50]:


#score = []
#for depth in range(5,20):
#    f1 = np.mean(cross_val_score(DecisionTreeClassifier(max_depth=depth), X_train_val, y_train_val, cv=kf, scoring='f1'))
#    score.append(f1)
#    print(depth)
#best_depth = list(range(5,20))[np.argmax(score)]  
#best_f1 = max(score)
#print('The best depth is {0} with f1-score of {1}'.format(best_depth,best_f1)) 


# In[51]:


Mean_AUC_tree_CV = np.mean(cross_val_score(DecisionTreeClassifier(max_depth=12), X_train_val, y_train_val, cv=kf, scoring='roc_auc'))
Mean_AUC_tree_CV 


# In[52]:


Mean_Precision_tree_CV = np.mean(cross_val_score(DecisionTreeClassifier(max_depth=12), X_train_val, y_train_val, cv=kf, scoring='precision'))
Mean_Precision_tree_CV 


# In[53]:


Mean_Recall_tree_CV = np.mean(cross_val_score(DecisionTreeClassifier(max_depth=12), X_train_val, y_train_val, cv=kf, scoring='recall'))
Mean_Recall_tree_CV 


# In[54]:


#score = []
#for depth in range(8,13):
#    f1 = np.mean(cross_val_score(RandomForestClassifier(max_depth=depth,random_state=42), X_train_val, y_train_val, cv=kf, scoring='f1'))
#    score.append(f1)
#    print(depth)
#best_depth = list(range(8,13))[np.argmax(score)]  
#best_f1 = max(score)
#print('The best depth is {0} with f1-score of {1}'.format(best_depth,best_f1)) 


# In[55]:


#parameters = {'n_estimators':[170,200,230],'max_depth':[10,15,17],'random_state':[42]}
#rf = RandomForestClassifier()
#clf = GridSearchCV(rf,parameters,scoring='f1')
#clf


# In[56]:


#clf.fit(X_train_val, y_train_val)
#clf.best_estimator_


# In[57]:


#clf.best_score_


# In[58]:


Mean_AUC_forest_CV = np.mean(cross_val_score(RandomForestClassifier(max_depth=17,random_state=42), X_train_val, y_train_val, cv=kf, scoring='roc_auc'))
Mean_AUC_forest_CV


# In[59]:


Mean_Precision_forest_CV = np.mean(cross_val_score(RandomForestClassifier(max_depth=17,random_state=42), X_train_val, y_train_val, cv=kf, scoring='precision'))
Mean_Precision_forest_CV


# In[60]:


Mean_Recall_forest_CV = np.mean(cross_val_score(RandomForestClassifier(max_depth=17,random_state=42), X_train_val, y_train_val, cv=kf, scoring='recall'))
Mean_Recall_forest_CV


# In[61]:


Log_Model = LogisticRegression(C=0.04138384,max_iter=10000)
KNN_Model = KNeighborsClassifier(n_neighbors=7)
NB_Model = GaussianNB()
Tree_Model = DecisionTreeClassifier(max_depth=12)
Forest_Model = RandomForestClassifier(max_depth=17,random_state=42)

model_list = [Log_Model,KNN_Model,NB_Model,Tree_Model,Forest_Model]
model_names = ["log_model", "knn_model", "nb_model", "tree_model", "forest_model"]
model = list(zip(model_names, model_list))


# In[62]:


Mean_AUC_ensemble_CV = np.mean(cross_val_score(VotingClassifier(estimators=model,voting='soft',n_jobs=-1), X_train_val, y_train_val, cv=kf, scoring='roc_auc'))
Mean_AUC_ensemble_CV


# In[63]:


Mean_Precision_ensemble_CV = np.mean(cross_val_score(VotingClassifier(estimators=model,voting='soft',n_jobs=-1), X_train_val, y_train_val, cv=kf, scoring='precision'))
Mean_Precision_ensemble_CV


# In[64]:


Mean_Recall_ensemble_CV = np.mean(cross_val_score(VotingClassifier(estimators=model,voting='soft',n_jobs=-1), X_train_val, y_train_val, cv=kf, scoring='recall'))
Mean_Recall_ensemble_CV


# In[65]:


model = ['Logistic Regression','KNN','Gaussian NB','Decision Trees','Random Forest','Ensemble']
scoring = ['AUC','Precision','Recall']
model_name = ['Logit','KNN','NB','tree','forest','ensemble']
model_list = []

for i in model:
    for j in scoring:
        model_dic = {'Model': i,'Scoring':j, 'Score':eval('Mean_{0}_{1}_CV'.format(j,model_name[model.index(i)]))}
        model_list.append(model_dic)


# In[66]:


model_df = pd.DataFrame(model_list)
model_df


# In[67]:


sns.set(style='white',font_scale=1)
fig = plt.figure(figsize=[12,9])
ax = sns.barplot(x='Model',y='Score',data=model_df,hue='Scoring',palette='Blues')
for p in ax.patches:
        ax.annotate(p.get_height().round(3), (p.get_x()+0.01, p.get_height()+0.001))
plt.legend(title='Score Metric',loc="upper right", bbox_to_anchor=(1.17, 0.5),fontsize=13)
plt.ylim([0.8,1.0])
plt.yticks([0.80,0.85,0.90,0.95,1.00])
plt.xlabel('',weight='bold',fontsize='15')
plt.ylabel('Score',weight='bold',fontsize='18')
plt.title('Mean Performance of CV Across Models',weight='bold',fontsize=20)
sns.despine()
plt.savefig('modelbarplot.png',transparent=True, bbox_inches='tight')


# In[68]:


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,random_state=42)


# In[69]:


rf = RandomForestClassifier(max_depth=17, random_state=42)
rf.fit(X_train,y_train)
print("Random Forest score: {:.4f}".format(rf.score(X_train,y_train)))


# In[70]:


y_predict = rf.predict_proba(X_val)[:, 1] >= 0.5
precision_05 = precision_score(y_val, y_predict)
precision_05


# In[71]:


recall_05=recall_score(y_val, y_predict)
recall_05


# In[72]:


y_predict = rf.predict_proba(X_val)[:, 1] >= 0.7
precision_07 = precision_score(y_val, y_predict)
precision_07


# In[73]:


recall_07=recall_score(y_val, y_predict)
recall_07


# In[74]:


roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])


# In[75]:


SV_model = [{'Scoring':'Precision','Threshold':0.5,'Score':precision_05},{'Scoring':'Precision','Threshold':0.7,'Score':precision_07},{'Scoring':'Recall','Threshold':0.5,'Score':recall_05},{'Scoring':'Recall','Threshold':0.7,'Score':recall_07}]
SV_model = pd.DataFrame(SV_model)
SV_model


# In[76]:


sns.set(style='white',font_scale=1.2)
fig = plt.figure(figsize=[7,7])
ax = sns.barplot(x='Scoring',y='Score',data=SV_model,hue='Threshold',palette='Purples')
for p in ax.patches:
        ax.annotate(p.get_height().round(3), (p.get_x()+0.1, p.get_height()+0.001))
plt.legend(title='Probability Threshold',loc="upper right", bbox_to_anchor=(0.95, 1),fontsize=13)
plt.ylim([0.8,1.0])
plt.xlabel('',weight='bold',fontsize='15')
plt.ylabel('Score',weight='bold',fontsize='18')
sns.despine()
plt.savefig('probabilityplot.png',transparent=True, bbox_inches='tight')


# # Model Evaluation
# 
# #### Model selected is Random Forest (depth=17) with threshold>=0.7. We evaluate the model on the test set:
# - Plotting the confusion matrix, ROC curve
# - Find out the precision, recall and AUC
# 
# #### The model is then tested on a business problem - How to ensure first-time customer satisfaction for economy/business:
# - Random Forest Feature Importance is plotted to understand which feature scores to adjust
# - Good Inflight Wifi service is crucial for customer satisfaction for both economy/business class customers
# - Ease of online booking is important to business class customers

# In[77]:


rf = RandomForestClassifier(max_depth=17,random_state=42)
rf.fit(X_train_val,y_train_val)
print("Random Forest score: {:.4f}".format(rf.score(X_train_val,y_train_val)))


# In[78]:


def make_confusion_matrix(model, threshold=0.7):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    satisfaction_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=120)
    ax = sns.heatmap(satisfaction_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['Neutral/Dissatisfied', 'Satisfied'],
           yticklabels=['Neutral/Dissatisfied', 'Satisfied']);   
    plt.xlabel('Prediction',weight='bold',fontsize=12)
    plt.ylabel('Actual',weight='bold',fontsize=12)
    plt.title('Confusion Matrix',weight='bold',fontsize=15)


# In[79]:


sns.set(style='white',font_scale=1)
make_confusion_matrix(rf)
plt.savefig('confusionplot.png',transparent=True, bbox_inches='tight')


# In[80]:


interactive(lambda threshold: make_confusion_matrix(rf, threshold), threshold=(0.0,1.0,0.02))


# In[81]:


y_predict = rf.predict_proba(X_test)[:, 1]>=0.7
precision_score(y_test, y_predict)


# In[82]:


recall_score(y_test, y_predict)


# In[83]:


f1_score(y_test, y_predict)


# In[84]:


roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])


# In[85]:


fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])


# In[86]:


sns.set(style='white',font_scale=1.2)
fig = plt.figure(figsize=[10,7])
plt.plot(fpr, tpr,lw=2,label='RandomForest (AUC={:.3f})'.format(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])))
plt.plot([0,1],[0,1],c='violet',ls='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.legend(loc="lower right",fontsize=15)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic (ROC) Curve',weight='bold',fontsize=15);
sns.despine()
plt.savefig('ROCplot.png',transparent=True, bbox_inches='tight')


# In[87]:


rf.feature_importances_


# In[88]:


df_impt = pd.DataFrame({'features':X_train_val.columns,'importance':rf.feature_importances_})
df_impt = df_impt.sort_values(by='importance',ascending=True)
df_impt


# In[89]:


sns.set(style='darkgrid',font_scale=1.2)
fig = plt.figure(figsize=[10,7])
ax = plt.barh(df_impt['features'],df_impt['importance'])
plt.title('Random Forest - Feature Importance',weight='bold',fontsize=16)
plt.savefig('featureplot.png',transparent=True, bbox_inches='tight')


# In[90]:


Test_Set = pd.DataFrame(y_test).join(X_test).join(pd.DataFrame(DF['Id']))
Test_Set['Predicted Satisfaction'] = (rf.predict_proba(X_test)[:, 1]>=0.7).astype(int)
Test_Set = Test_Set[['Satisfaction','Predicted Satisfaction','Inflight Wifi Service','Ease Of Online Booking','Food And Drink','Online Boarding','Seat Comfort','Inflight Entertainment','On-board Service','Leg Room','Baggage Handling','Checkin Service','Inflight Service','Cleanliness','Customer Type_Returning Customer','Type Of Travel_Personal Travel','Class_Economy']]
Test_Set.reset_index(inplace=True)
Test_Set.drop('index',axis=1,inplace=True)
Test_Set['Satisfaction'] = Test_Set['Satisfaction'].map({0:'Neutral/Dissatisfied',1:'Satisfied'})
Test_Set['Predicted Satisfaction'] = Test_Set['Predicted Satisfaction'].map({0:'Neutral/Dissatisfied',1:'Satisfied'})
Test_Set


# In[91]:


Test_Set[(Test_Set['Class_Economy']==0)&(Test_Set['Satisfaction']=='Satisfied')&(Test_Set['Type Of Travel_Personal Travel']==0)&((Test_Set['Inflight Wifi Service']!=5))].head(100)


# In[92]:


Test_Set_Economy = Test_Set[(Test_Set['Class_Economy']==1)&(Test_Set['Type Of Travel_Personal Travel']==1)&(Test_Set['Customer Type_Returning Customer']==0)]
Test_Set_Economy['Satisfaction'] = Test_Set_Economy['Satisfaction'].map({'Neutral/Dissatisfied':0,'Satisfied':1})
Test_Set_Economy['Predicted Satisfaction'] = Test_Set_Economy['Predicted Satisfaction'].map({'Neutral/Dissatisfied':0,'Satisfied':1})
precision_score(Test_Set_Economy['Satisfaction'], Test_Set_Economy['Predicted Satisfaction'])


# In[93]:


## NEW ECONOMY (Personal Travel) CUSTOMERS ##
neweconomy = []
neweconomy.append({'Inflight Wifi Service':3,'Ease Of Online Booking':3,'Food And Drink':3,'Online Boarding':3,'Seat Comfort':3,'Inflight Entertainment':3,'On-board Service':3,'Leg Room':3,'Baggage Handling':3,'Checkin Service':3,'Inflight Service':3,'Cleanliness':3,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':1,'Class_Economy':1})     
neweconomy.append({'Inflight Wifi Service':5,'Ease Of Online Booking':3,'Food And Drink':3,'Online Boarding':3,'Seat Comfort':3,'Inflight Entertainment':3,'On-board Service':3,'Leg Room':3,'Baggage Handling':3,'Checkin Service':3,'Inflight Service':3,'Cleanliness':3,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':1,'Class_Economy':1})
neweconomy.append({'Inflight Wifi Service':4,'Ease Of Online Booking':5,'Food And Drink':5,'Online Boarding':5,'Seat Comfort':5,'Inflight Entertainment':5,'On-board Service':5,'Leg Room':5,'Baggage Handling':5,'Checkin Service':5,'Inflight Service':5,'Cleanliness':5,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':1,'Class_Economy':1})
neweconomy.append({'Inflight Wifi Service':3,'Ease Of Online Booking':5,'Food And Drink':5,'Online Boarding':5,'Seat Comfort':5,'Inflight Entertainment':5,'On-board Service':5,'Leg Room':5,'Baggage Handling':5,'Checkin Service':5,'Inflight Service':5,'Cleanliness':5,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':1,'Class_Economy':1}) 
neweconomy = pd.DataFrame(neweconomy)


# In[94]:


neweconomy['Predicted Satisfaction'] = (rf.predict_proba(neweconomy)[:, 1]>=0.7).astype(int)
neweconomy = neweconomy[['Predicted Satisfaction','Inflight Wifi Service','Ease Of Online Booking','Food And Drink','Online Boarding','Seat Comfort','Inflight Entertainment','On-board Service','Leg Room','Baggage Handling','Checkin Service','Inflight Service','Cleanliness','Customer Type_Returning Customer','Type Of Travel_Personal Travel','Class_Economy']]
neweconomy['Predicted Satisfaction'] = neweconomy['Predicted Satisfaction'].map({0:'Neutral/Dissatisfied',1:'Satisfied'})
neweconomy


# In[95]:


Test_Set_Business = Test_Set[(Test_Set['Class_Economy']==0)&(Test_Set['Type Of Travel_Personal Travel']==0)&(Test_Set['Customer Type_Returning Customer']==0)]
Test_Set_Business['Satisfaction'] = Test_Set_Business['Satisfaction'].map({'Neutral/Dissatisfied':0,'Satisfied':1})
Test_Set_Business['Predicted Satisfaction'] = Test_Set_Business['Predicted Satisfaction'].map({'Neutral/Dissatisfied':0,'Satisfied':1})
precision_score(Test_Set_Business['Satisfaction'], Test_Set_Business['Predicted Satisfaction'])


# In[96]:


## NEW Business (Business Travel) CUSTOMERS ##
newbusiness = []
newbusiness.append({'Inflight Wifi Service':3,'Ease Of Online Booking':3,'Food And Drink':3,'Online Boarding':3,'Seat Comfort':3,'Inflight Entertainment':3,'On-board Service':3,'Leg Room':3,'Baggage Handling':3,'Checkin Service':3,'Inflight Service':3,'Cleanliness':3,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':0,'Class_Economy':0})     
newbusiness.append({'Inflight Wifi Service':5,'Ease Of Online Booking':3,'Food And Drink':3,'Online Boarding':3,'Seat Comfort':3,'Inflight Entertainment':3,'On-board Service':3,'Leg Room':3,'Baggage Handling':3,'Checkin Service':3,'Inflight Service':3,'Cleanliness':3,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':0,'Class_Economy':0})
newbusiness.append({'Inflight Wifi Service':3,'Ease Of Online Booking':5,'Food And Drink':5,'Online Boarding':5,'Seat Comfort':5,'Inflight Entertainment':5,'On-board Service':5,'Leg Room':5,'Baggage Handling':5,'Checkin Service':5,'Inflight Service':5,'Cleanliness':5,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':0,'Class_Economy':0})
newbusiness.append({'Inflight Wifi Service':3,'Ease Of Online Booking':4,'Food And Drink':4,'Online Boarding':4,'Seat Comfort':4,'Inflight Entertainment':4,'On-board Service':4,'Leg Room':4,'Baggage Handling':4,'Checkin Service':4,'Inflight Service':4,'Cleanliness':4,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':0,'Class_Economy':0})
newbusiness.append({'Inflight Wifi Service':3,'Ease Of Online Booking':5,'Food And Drink':4,'Online Boarding':4,'Seat Comfort':4,'Inflight Entertainment':4,'On-board Service':4,'Leg Room':4,'Baggage Handling':4,'Checkin Service':4,'Inflight Service':4,'Cleanliness':4,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':0,'Class_Economy':0})
newbusiness.append({'Inflight Wifi Service':3,'Ease Of Online Booking':4,'Food And Drink':5,'Online Boarding':5,'Seat Comfort':5,'Inflight Entertainment':5,'On-board Service':5,'Leg Room':5,'Baggage Handling':5,'Checkin Service':5,'Inflight Service':5,'Cleanliness':5,'Customer Type_Returning Customer':0,'Type Of Travel_Personal Travel':0,'Class_Economy':0})
newbusiness = pd.DataFrame(newbusiness)


# In[97]:


newbusiness['Predicted Satisfaction'] = (rf.predict_proba(newbusiness)[:, 1]>=0.5).astype(int)
newbusiness = newbusiness[['Predicted Satisfaction','Inflight Wifi Service','Ease Of Online Booking','Food And Drink','Online Boarding','Seat Comfort','Inflight Entertainment','On-board Service','Leg Room','Baggage Handling','Checkin Service','Inflight Service','Cleanliness','Customer Type_Returning Customer','Type Of Travel_Personal Travel','Class_Economy']]
newbusiness['Predicted Satisfaction'] = newbusiness['Predicted Satisfaction'].map({0:'Neutral/Dissatisfied',1:'Satisfied'})
newbusiness


# In[104]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create a KNN classifier with desired parameters
k = 3  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels for training set
y_train_pred = knn.predict(X_train)

# Calculate accuracy on training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on KNeighborsClassifier training set: {:.2f}".format(train_accuracy))

# Predict the labels for testing set
y_test_pred = knn.predict(X_test)

# Calculate accuracy on testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on KNeighborsClassifier testing set: {:.2f}".format(test_accuracy))


# In[106]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a logistic regression classifier with desired parameters
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels for training set
y_train_pred = logreg.predict(X_train)

# Calculate accuracy on training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on LogisticRegression training set: {:.2f}".format(train_accuracy))

# Predict the labels for testing set
y_test_pred = logreg.predict(X_test)

# Calculate accuracy on testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on LogisticRegression testing set: {:.2f}".format(test_accuracy))


# In[105]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a Random Forest classifier with desired parameters
rf = RandomForestClassifier()

# Fit the classifier to the training data
rf.fit(X_train, y_train)

# Predict the labels for training set
y_train_pred = rf.predict(X_train)

# Calculate accuracy on training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on RandomForestClassifier training set: {:.2f}".format(train_accuracy))

# Predict the labels for testing set
y_test_pred = rf.predict(X_test)

# Calculate accuracy on testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on RandomForestClassifier testing set: {:.2f}".format(test_accuracy))


# In[107]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a Decision Tree classifier with desired parameters
dt = DecisionTreeClassifier()

# Fit the classifier to the training data
dt.fit(X_train, y_train)

# Predict the labels for training set
y_train_pred = dt.predict(X_train)

# Calculate accuracy on training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on DecisionTreeClassifier training set: {:.2f}".format(train_accuracy))

# Predict the labels for testing set
y_test_pred = dt.predict(X_test)

# Calculate accuracy on testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on DecisionTreeClassifier testing set: {:.2f}".format(test_accuracy))


# In[113]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load your training data and split into features (X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the training set and testing set
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

# Calculate accuracy on the training set and testing set
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

# Print accuracy
print(f"Accuracy on training set: {accuracy_train:.2f}")
print(f"Accuracy on testing set: {accuracy_test:.2f}")

# Create confusion matrix for training set and testing set
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix for training set
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix (Training Set)")
plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot confusion matrix for testing set
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix (Testing Set)")
plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:




