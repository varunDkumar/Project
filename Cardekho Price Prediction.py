#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


data= pd.read_csv('car data.csv')


# In[24]:


data.head()


# In[25]:


data.info()


# In[26]:


data.shape


# In[27]:


f=['Seller_Type','Transmission','Owner','Fuel_Type']
for i in f:
    print(data[i].unique())


# In[28]:


data.isnull().sum()


# In[29]:


data.describe()


# In[30]:


data.columns


# In[31]:


dataset= data[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Seller_Type','Fuel_Type','Transmission', 'Owner']]


# In[32]:


dataset['currentyear']=2021
dataset.head()


# In[33]:


dataset['yearsused']= dataset['currentyear']-dataset['Year']
dataset.head()


# In[34]:


dataset.drop(['Year'],axis=1, inplace=True)
dataset.drop(['currentyear'],axis=1,inplace=True)
dataset.head()


# In[35]:


dataset= pd.get_dummies(dataset,drop_first=True)
dataset.head()


# In[36]:


dataset.corr()


# In[37]:


mat=dataset.corr()
feat=mat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(dataset[feat].corr(),annot=True)


# In[38]:


x=dataset.iloc[:,1:]
y=dataset.iloc[:,0]
x.head()


# In[39]:


y.head()


# In[40]:


from sklearn.ensemble import ExtraTreesRegressor
mod=ExtraTreesRegressor()
mod.fit(x,y)


# In[41]:


print(mod.feature_importances_)


# In[42]:


graph= pd.Series(mod.feature_importances_, index=x.columns)
graph.nlargest(5).plot(kind='barh')
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_test


# In[44]:


from sklearn.ensemble import RandomForestRegressor
rand=RandomForestRegressor()


# In[45]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[46]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[47]:


rf = RandomForestRegressor()


# In[48]:


from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[49]:


rf_random.fit(x_train,y_train)


# In[50]:


pred=rf_random.predict(x_test)
pred


# In[51]:


sns.distplot(y_test-pred)


# In[52]:


plt.scatter(y_test,pred)


# In[53]:


rf_random.predict([[7.6,28000,0,1,0,0,0,0]])


# In[ ]:




