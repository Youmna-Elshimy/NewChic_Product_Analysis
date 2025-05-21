#!/usr/bin/env python
# coding: utf-8

# In[49]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler





import warnings
warnings.filterwarnings('ignore')





#importing csv files
men = pd.read_csv('men.csv')
women = pd.read_csv('women.csv')
bags = pd.read_csv('bags.csv')
beauty = pd.read_csv('beauty.csv')
house = pd.read_csv('house.csv')
jewelry = pd.read_csv('jewelry.csv')
accessories = pd.read_csv('accessories.csv')
kids = pd.read_csv('kids.csv')
shoes = pd.read_csv('shoes.csv')





dataframes = [men,women,bags,kids,beauty,house,jewelry,accessories,shoes]





#combining 9 categories
combined_dataframe = pd.concat(dataframes,ignore_index=True)





columns_to_keep = ['current_price','name', 'raw_price', 'discount', 'likes_count', 'is_new', 'category', 'subcategory', 'brand','codCountry']





updated_dataframe = combined_dataframe[columns_to_keep]




#brand column has 60k missing values, hence we are dropping it 
updated_dataframe = updated_dataframe.drop('brand',axis=1)




updated_dataframe['is_new'] = updated_dataframe['is_new'].astype(int)





#filling values of 'codCountry' column





updated_dataframe['codCountry'].fillna('ID,MY,PH,SG,TH,VN', inplace=True)





#dropping duplicates
updated_dataframe.drop_duplicates(inplace=True)





#dropping rows which have missing values
updated_dataframe.dropna(inplace=True)





#ensuring all rows are unique
unique_instances = updated_dataframe.drop_duplicates()
print(f"Number of unique rows: {unique_instances.shape[0]}")
print(f"Total rows in the dataset: {updated_dataframe.shape[0]}")





#detecting inconsistencies
if unique_instances.shape[0] != updated_dataframe.shape[0]:
    print("There are duplicate rows in the dataset.")
else:
    print("All rows are unique.")





#applying scaling/normalization to remove bias
scaler = MinMaxScaler()
updated_dataframe['likes_count'] = scaler.fit_transform(updated_dataframe[['likes_count']])


# tested the action of making raw price as current price if discount is 0, on a subset of original dataframe. applying the same technique on original dataframe




updated_dataframe.loc[updated_dataframe['discount'] == 0, ['raw_price']] = updated_dataframe['current_price']


print(updated_dataframe.describe())


updated_dataframe.to_csv('Updated_CSV', index=False)


# Visuals

# ![Alt Text](1.jpg)
#

# ![Alt Text](2.jpg)
#

# ![Alt Text](3.jpg)
#

# ![Alt Text](4.jpg)
#

# In[67]:


df_numeric = updated_dataframe.select_dtypes(include=['number'])


# In[68]:



plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='inferno', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
