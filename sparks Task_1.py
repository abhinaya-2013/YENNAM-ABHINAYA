#!/usr/bin/env python
# coding: utf-8

# # Yennam Abhinaya        TASK-1

# # Loading Libraries

# In[46]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Dataset

# In[47]:


data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print("Data imported successfully")
data


# # Understanding Data

# In[48]:


data.head()


# # checking for Null Values

# In[49]:


data.isnull().sum()


# # checking for duplicate values

# In[50]:


data.duplicated().sum()


# In[51]:


data.shape


# In[52]:


data.describe()


# In[53]:


data.info()


# # Data Analysis

# In[54]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Training and Testing Data

# In[55]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# # Applying Linear Regression Algorithm

# In[56]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Training the Algorithm

# In[57]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print("Training complete.")


# In[58]:


print('Coefficients: \n', lm.coef_)


# # Plotting the Regression Line

# In[59]:


# Plotting the regression line
line = lm.coef_*X+lm.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # Making the Prediction

# In[60]:


print(X_test) # Testing data - In Hours
y_pred = lm.predict(X_test) # Predicting the scores


# In[61]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[62]:


prediction = lm.predict(X_test)


# In[63]:


plt.scatter(y_test,prediction)
plt.xlabel('xlabel')
plt.ylabel('ylabel')


# # Evaluating the Model

# In[64]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[66]:


hours = [[9.25]]
pred = lm.predict(hours)


# In[67]:


pred


# In[68]:


from sklearn.metrics import r2_score
print('r2 score:' , metrics.r2_score(y_test,prediction))


# In[ ]:




