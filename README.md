# EXNO:4-DS
# NAME:VIMALA SAHANA W
# REG NO:212223040241
# DATE:26-04-25
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/586e05c5-ac84-4237-9e86-bc5200ab3bfb)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/548bc747-7a17-41ad-8b2b-148faa9122d5)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/ee906889-d8ac-4727-81b6-c681a3856570)
```
max_vals=np.max(np.abs(df[['Height','Weight']]),axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/1e417587-31b8-4871-8579-eca5f9f45c6d)
```
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/3a7db33a-7d77-4d9f-bd37-d153905a2e1c)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/915bce04-a3c3-4877-9dc5-7ce6ac257afa)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1=pd.read_csv("/content/bmi.csv")
df1.head()
df1[['Height','Weight']] = scaler.fit_transform(df1[['Height','Weight']])
df1
```
![image](https://github.com/user-attachments/assets/5f933cef-bfe2-4b7e-883d-90905bb45090)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/785db26f-e20d-42c4-920d-de7adedcc803)
```
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/dee65e62-b0bc-46cf-93c5-9ec1f27a47cc)
```
df
```
![image](https://github.com/user-attachments/assets/395fdcd6-a973-43b2-a3e0-623655f044b5)
```
df.info()
```
![image](https://github.com/user-attachments/assets/bed6503e-cb08-4ec6-a427-b3dc12d07fa4)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/06d2bc9d-54d7-4463-a159-fa6e124dcf67)
```
categorical_columns=['JobType','EdType','maritalstatus','occupation','relationship','race','gender','nativecountry']
df[categorical_columns]=df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/8c5d8dc4-67e3-4d3f-817c-2bef8ba3187c)
```
df[categorical_columns]=df[categorical_columns].apply(lambda x:x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ce9c9c13-b742-4b53-8d6b-604fe31a98f9)
```
x=df.drop(columns=['SalStat'])
y=df['SalStat']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(f"Modeel accuracy using selected features: {acc}")
```
![image](https://github.com/user-attachments/assets/98a2f3ef-56ad-4202-a788-eb7f8f226570)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,chi2,f_classif
categorical_columns=['JobType','EdType','maritalstatus','occupation','relationship','race','gender','nativecountry']
df[categorical_columns]=df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/5179b562-b55c-414f-a4f8-f59c9d3cdccd)
```
df[categorical_columns]=df[categorical_columns].apply(lambda x:x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/e65e171c-75de-40eb-b87e-3d220651a63b)
```
x=df.drop(columns=['SalStat'])
y=df['SalStat']
k_chi2=6
selector_chi2=SelectKBest(score_func=chi2,k=k_chi2)
x_chi2=selector_chi2.fit_transform(x,y)
selected_features_chi2=x.columns[selector_chi2.get_support()]
print("Selected Features (Chi-squared):",selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/de90d46a-f0d5-4567-8f53-b2b537240b53)
```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
       'hoursperweek']
x=df[selected_features]
y=df['SalStat']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(f"Model accuracy using selected features: {acc}")
```
![image](https://github.com/user-attachments/assets/7d6dda2a-4bc2-40e9-ba4c-e61cb9049e6c)
```
# @title
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/938366bd-c3e1-4b2d-82c6-10ead17ad663)
```
# @title 
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

categorical_columns=['JobType','EdType','maritalstatus','occupation','relationship','race','gender','nativecountry']
df[categorical_columns]=df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4785a9e5-7ac3-4ec7-b93e-7fae5ca4f792)
```
df[categorical_columns]=df[categorical_columns].apply(lambda x:x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4bdb6381-9df1-42f6-b22a-37e3809a25c6)
```
k_anova=5
selector_anova=SelectKBest(score_func=f_classif,k=k_anova)
x_anova=selector_anova.fit_transform(x,y)
selected_features_anova=x.columns[selector_anova.get_support()]
print("Selected Features (ANOVA):",selected_features)
```
![image](https://github.com/user-attachments/assets/4011e492-07a5-41aa-b401-0e17492fd3bb)
```
 import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/df24a5b3-76a6-4ea3-a6b6-591f332db1b1)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![Screenshot 2025-04-26 113323](https://github.com/user-attachments/assets/b007a0d8-9187-4857-b095-bda97a26c37d)
```
x=df.drop(columns=['SalStat'])
y=df['SalStat']
logreg=LogisticRegression()
n_features_to_select=6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(x, y)
```


![Screenshot 2025-04-26 113508](https://github.com/user-attachments/assets/a6acdbdc-8657-4b76-b954-06454339bd4b)

![image](https://github.com/user-attachments/assets/7ea8c574-7516-4e59-a5ea-09dac8404e1d)
```
selected_features = x.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/be58c21c-b880-4db4-a88c-f43fd0df3a68)
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
x_selected = x[selected_features]
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```

![image](https://github.com/user-attachments/assets/12a5206c-be8d-41f4-a5b5-daa7c4a71132)









# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
