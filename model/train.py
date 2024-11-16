# %%
#import libraries
import pandas as pd
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.metrics import r2_score

# %%
data = pd.read_csv("Chennai.csv")

# %%
data.info()

# %%
data.describe()

# %%
data.columns

# %%
sns.heatmap(data.corr(), annot=True)

# %%
X = data[['Price', 'Area', 'Location', 'No. of Bedrooms', 'Resale',
       'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
       'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
       'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School',
       '24X7Security', 'PowerBackup', 'CarParking', 'StaffQuarter',
       'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
       'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 'LiftAvailable',
       'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
       'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator']]

y = data['Price']

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# %%
X_train

# %%
K=np.log(X_train['Price']/X_train['Area'])

plt.plot(K,X_train['Price'],'o')
plt.xlabel('Location decoded value')
plt.ylabel('Price')
plt.title('Bangalore housing price')
plt.show()


# %%
matrix_corr=X_train.corr()
matrix_corr['Price']

# %%
def na_remove(data):
    data.replace(9,0.5,inplace=True)

# %%
def data_processing(data):
    K=np.log(data['Price']/data['Area'])
    data['Location']=K
    house_feature=data.drop(['Price'],axis=1)
    my_pipeline=Pipeline([('rem',na_remove(house_feature)),
                          ('std',StandardScaler())   
                         ])
    return my_pipeline.fit_transform(house_feature)

# %%
houseprice_train=np.log(X_train['Price'])
data_train=data_processing(X_train)

# %%
data_train.shape

# %% [markdown]
# # Different model training

# %%
model1=LinearRegression().fit(data_train,houseprice_train)

# %%
model2=Ridge().fit(data_train,houseprice_train)

# %%
model3=DecisionTreeRegressor().fit(data_train,houseprice_train)

# %%
model4=RandomForestRegressor().fit(data_train,houseprice_train)

# %%
model1_pred=model1.predict(data_train)
model2_pred=model2.predict(data_train)
model3_pred=model3.predict(data_train)
model4_pred=model4.predict(data_train)

# %%
sns.regplot(model1_pred,houseprice_train)

# %%
sns.regplot(model2_pred,houseprice_train)

# %%
sns.regplot(model3_pred,houseprice_train)

# %%
sns.regplot(model4_pred,houseprice_train)

# %% [markdown]
# # Testing

# %%
houseprice_test=X_test['Price']

# %%
data_test=data_processing(X_test)

# %%
model1_test=np.exp(model1.predict(data_test))
model2_test=np.exp(model2.predict(data_test))
model3_test=np.exp(model3.predict(data_test))
model4_test=np.exp(model4.predict(data_test))

# %%
sns.regplot(model1_test,houseprice_test)

# %%
sns.regplot(model2_test,houseprice_test)

# %%
sns.regplot(model3_test,houseprice_test)

# %%
sns.regplot(model4_test,houseprice_test)

# %%
model1_r2=r2_score(model1_test,houseprice_test)

model2_r2=r2_score(model2_test,houseprice_test)

model3_r2=r2_score(model3_test,houseprice_test)

model4_r2=r2_score(model4_test,houseprice_test)

# %%
print("model1_error:{}\nmodel2_error:{}\nmodel3_error:{}\nmodel4_error:{}".format(model1_r2,model2_r2,model3_r2,model4_r2))

# %% [markdown]
# # Model exporting

# %%
import joblib as jb

# %%
jb.dump(model4,'Chennai_house_price.pkl')

# %%



