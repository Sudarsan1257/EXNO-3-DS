## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
**STEP 1:Read the given Data.**

**STEP 2:Clean the Data Set using Data Cleaning Process.**

**STEP 3:Apply Feature Encoding for the feature in the data set.**

**STEP 4:Apply Feature Transformation for the feature in the data set.**

**STEP 5:Save the data to the file.**

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
ds=pd.read_csv("Encoding Data.csv")
ds
```
<img width="534" height="491" alt="image" src="https://github.com/user-attachments/assets/9a67ce3c-ce86-4591-b821-685da3016f40" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
a=['Hot','Warm','Cold']
a1=OrdinalEncoder(categories=[a])
a1.fit_transform(ds[["ord_2"]])
```

<img width="706" height="327" alt="image" src="https://github.com/user-attachments/assets/bb27ba28-fea3-4bd6-a186-ec697427b442" />


```
ds['bo2']=a1.fit_transform(ds[["ord_2"]])
ds
```
<img width="871" height="493" alt="image" src="https://github.com/user-attachments/assets/1abe2187-1465-4b44-8b20-9430e1ef9d0f" />

```
l=LabelEncoder()
dfc=ds.copy()
dfc['ord_2']=l.fit_transform(dfc['ord_2'])
dfc
```
<img width="611" height="519" alt="image" src="https://github.com/user-attachments/assets/510de76e-348d-4f65-b607-b72c202a9f50" />

```
from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(sparse_output=False)
ds2=ds.copy()
enc=pd.DataFrame(oh.fit_transform(ds2[["nom_0"]]))
ds2=pd.concat([ds2,enc],axis=1)
ds2
```
<img width="703" height="566" alt="image" src="https://github.com/user-attachments/assets/94910b04-2948-4089-94c1-cf16b3359995" />

```
pd.get_dummies(ds2,columns=["nom_0"])
```
<img width="899" height="479" alt="image" src="https://github.com/user-attachments/assets/178c0bcd-2d1a-4fdd-957f-c075ee90dfb6" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="807" height="504" alt="image" src="https://github.com/user-attachments/assets/fff6961c-5f70-4204-8724-0368f42934c6" />

```
b=BinaryEncoder()
nd=b.fit_transform(df['Ord_2'])
df
```
<img width="744" height="507" alt="image" src="https://github.com/user-attachments/assets/675a645f-bfd0-4814-b247-3fda778c3f07" />

```
db=pd.concat([df,nd],axis=1)
db
```
<img width="885" height="482" alt="image" src="https://github.com/user-attachments/assets/80a01502-5f46-4a08-a9e0-a2a4396383d4" />

```
from category_encoders import TargetEncoder
t=TargetEncoder()
CC=df.copy()
new=t.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="889" height="568" alt="image" src="https://github.com/user-attachments/assets/6c894e9d-2c64-4c69-a8a0-8a66fd1bd1db" />

```
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
<img width="1108" height="589" alt="image" src="https://github.com/user-attachments/assets/8cd9c518-6fed-4f2a-9fad-eb4857ea1d3b" />

```
df.skew()
```
<img width="597" height="304" alt="image" src="https://github.com/user-attachments/assets/9b23981d-69d7-4752-83ca-d847d6baae34" />

```
np.log(df["Highly Positive Skew"])
```
<img width="692" height="570" alt="image" src="https://github.com/user-attachments/assets/0b969fd7-d105-48ee-bf6d-809931f59b9a" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="819" height="567" alt="image" src="https://github.com/user-attachments/assets/4799cda9-0008-44d4-bd82-b96f5e9880ac" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="667" height="574" alt="image" src="https://github.com/user-attachments/assets/934c9b27-b4b3-4bc6-8f14-c712016161e8" />

```
np.square(df["Highly Positive Skew"])
```
<img width="735" height="569" alt="image" src="https://github.com/user-attachments/assets/a017131b-6bd7-48da-8054-7ba7c8fb4d6b" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1377" height="554" alt="image" src="https://github.com/user-attachments/assets/14c1c2bb-061c-42f1-866e-fea53cadf865" />

```
df.skew()
```
<img width="689" height="340" alt="image" src="https://github.com/user-attachments/assets/9e865694-711b-4baf-ade4-448df8f2a597" />

```
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
<img width="1199" height="383" alt="image" src="https://github.com/user-attachments/assets/c811148a-07ce-4993-9262-59d862694518" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1717" height="617" alt="image" src="https://github.com/user-attachments/assets/8eecd93b-528c-409a-add8-272703274c9f" />


```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="1051" height="670" alt="image" src="https://github.com/user-attachments/assets/b378fadf-72bf-4991-bbda-149f29eca7b5" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
<img width="967" height="583" alt="image" src="https://github.com/user-attachments/assets/773c0ba7-80c3-4736-8547-30e5d66b6608" />

```
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="1058" height="620" alt="image" src="https://github.com/user-attachments/assets/563f7fc3-1512-42d6-b60b-8bec23dfaf88" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="945" height="601" alt="image" src="https://github.com/user-attachments/assets/6d1ba85f-266c-4248-bfcf-7a32b9abac83" />

```
df1=pd.read_csv("titanic_dataset.csv")
df1
```
<img width="1608" height="549" alt="image" src="https://github.com/user-attachments/assets/7e11e018-6bd6-4737-bba8-d3e65e47b07f" />

```
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df1["Age_1"]=qt.fit_transform(df1[["Age"]])
sm.qqplot(df1['Age'],line='45')
plt.show()
```
<img width="938" height="639" alt="image" src="https://github.com/user-attachments/assets/8b271fb3-8431-43e3-855f-9ce1cd62c79c" />

```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()
```
<img width="809" height="567" alt="image" src="https://github.com/user-attachments/assets/2416b0ab-a5ab-4fdb-96ef-2fd6d7290317" />


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
