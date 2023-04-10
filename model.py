import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings                    #import waarnigs to ingonre any kind of warning while runing code
warnings.filterwarnings("ignore")

# Load the dataset
data= pd.read_csv('client_data.csv')
print(data.head())

# Check missing values
print(data.isnull().sum())

# Drop the rows with missing values
data.dropna(inplace=True)

c25=data['cons_12m'].quantile(0.25)
c75=data["cons_12m"].quantile(0.75)
ciqr=c75-c25
cul=c75+1.5*ciqr
cll=c25-1.5*ciqr
data['cons_12m']=np.where(data['cons_12m']>cul,cul,np.where(data['cons_12m']<cll,cll,data['cons_12m']))

c25=data['cons_gas_12m'].quantile(0.25)
c75=data["cons_gas_12m"].quantile(0.75)
ciqr=c75-c25
cul=c75+1.5*ciqr
cll=c25-1.5*ciqr
data['cons_gas_12m']=np.where(data['cons_gas_12m']>cul,cul,np.where(data['cons_gas_12m']<cll,cll,data['cons_gas_12m']))

c25 = data['net_margin'].quantile(0.25)
c75 = data['net_margin'].quantile(0.75)
ciqr = c75 - c25
cul = c75 + 1.5 * ciqr
cll = c25 - 1.5 * ciqr
data['net_margin'] = np.where(data['net_margin'] > cul, cul, np.where(data['net_margin'] < cll, cll, data['net_margin']))

c25 = data['margin_gross_pow_ele'].quantile(0.25)
c75 = data['margin_gross_pow_ele'].quantile(0.75)
ciqr = c75 - c25
cul = c75 + 1.5 * ciqr
cll = c25 - 1.5 * ciqr
data['margin_gross_pow_ele'] = np.where(data['margin_gross_pow_ele'] > cul, cul, np.where(data['margin_gross_pow_ele'] < cll, cll, data['margin_gross_pow_ele']))

# Replace outliers with upper and lower limits
c25 = data['cons_last_month'].quantile(0.25)
c75 = data['cons_last_month'].quantile(0.75)
ciqr = c75 - c25
cul = c75 + 1.5 * ciqr
cll = c25 - 1.5 * ciqr
data['cons_last_month'] = np.where(data['cons_last_month'] > cul, cul, np.where(data['cons_last_month'] < cll, cll, data['cons_last_month']))

c25 = data['forecast_cons_12m'].quantile(0.25)
c75 = data['forecast_cons_12m'].quantile(0.75)
ciqr = c75 - c25
cul = c75 + 1.5 * ciqr
cll = c25 - 1.5 * ciqr
data['forecast_cons_12m'] = np.where(data['forecast_cons_12m'] > cul, cul, np.where(data['forecast_cons_12m'] < cll, cll, data['forecast_cons_12m']))

c25 = data['forecast_cons_year'].quantile(0.25)
c75 = data['forecast_cons_year'].quantile(0.75)
ciqr = c75 - c25
cul = c75 + 1.5 * ciqr
cll = c25 - 1.5 * ciqr
data['forecast_cons_year'] = np.where(data['forecast_cons_year'] > cul, cul, np.where(data['forecast_cons_year'] < cll, cll, data['forecast_cons_year']))

# Apply outlier treatment to 'forecast_meter_rent_12m'
q25 = data['forecast_meter_rent_12m'].quantile(0.25)
q75 = data['forecast_meter_rent_12m'].quantile(0.75)
iqr = q75 - q25
upper_limit = q75 + 1.5*iqr
lower_limit = q25 - 1.5*iqr
data['forecast_meter_rent_12m'] = np.where(data['forecast_meter_rent_12m'] > upper_limit, upper_limit, np.where(data['forecast_meter_rent_12m'] < lower_limit, lower_limit, data['forecast_meter_rent_12m']))

# Apply outlier treatment to 'forecast_price_energy_off_peak'
q25 = data['forecast_price_energy_off_peak'].quantile(0.25)
q75 = data['forecast_price_energy_off_peak'].quantile(0.75)
iqr = q75 - q25
upper_limit = q75 + 1.5*iqr
lower_limit = q25 - 1.5*iqr
data['forecast_price_energy_off_peak'] = np.where(data['forecast_price_energy_off_peak'] > upper_limit, upper_limit, np.where(data['forecast_price_energy_off_peak'] < lower_limit, lower_limit, data['forecast_price_energy_off_peak']))

# Apply outlier treatment to 'forecast_discount_energy'
q25 = data['forecast_discount_energy'].quantile(0.25)
q75 = data['forecast_discount_energy'].quantile(0.75)
iqr = q75 - q25
upper_limit = q75 + 1.5*iqr
lower_limit = q25 - 1.5*iqr
data['forecast_discount_energy'] = np.where(data['forecast_discount_energy'] > upper_limit, upper_limit, np.where(data['forecast_discount_energy'] < lower_limit, lower_limit, data['forecast_discount_energy']))

# Preprocessing
data['date_activ'] = pd.to_datetime(data['date_activ'], format='%Y-%m-%d')
data['date_end'] = pd.to_datetime(data['date_end'], format='%Y-%m-%d')
data['date_modif_prod'] = pd.to_datetime(data['date_modif_prod'], format='%Y-%m-%d')
data['date_renewal'] = pd.to_datetime(data['date_renewal'], format='%Y-%m-%d')

data['has_gas'] = data['has_gas'].map({'f': 0, 't': 1})

data['days_active'] = (pd.to_datetime('2016-01-01') - data['date_activ']).dt.days
data['days_since_modification'] = (pd.to_datetime('2016-01-01') - data['date_modif_prod']).dt.days
data['days_until_renewal'] = (data['date_renewal'] - pd.to_datetime('2016-01-01')).dt.days

data.drop(['id',  'channel_sales', 'date_activ', 'date_end', 'date_modif_prod', 'date_renewal', 'forecast_discount_energy', 'forecast_meter_rent_12m', 'forecast_price_energy_off_peak', 'forecast_price_energy_peak', 'forecast_price_pow_off_peak', 'nb_prod_act', 'origin_up', 'pow_max'], axis=1, inplace=True)

data.drop(['cons_gas_12m'],axis=1,inplace=True)

data['cons_12m']=data['cons_12m'].replace(0,data['cons_12m'].mean())
data['cons_last_month']=data['cons_last_month'].replace(0,data['cons_last_month'].mean())
data['forecast_cons_year']=data['forecast_cons_year'].replace(0,data['forecast_cons_year'].mean())
data['imp_cons']=data['imp_cons'].replace(0,data['imp_cons'].mean())
data['forecast_cons_12m']=data['forecast_cons_12m'].replace(0,data['forecast_cons_12m'].mean())





# Select independent and dependent variable
x = data[["forecast_cons_12m","cons_12m","imp_cons"]]

y = data['churn']
# Split the dataset into train and test

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=41)

# Feature scaling
gbc = GradientBoostingClassifier()
gbc.fit(xtrain, ytrain)

# Get predicted probabilities
ypred_proba = gbc.predict_proba(xtest)

# Set new threshold
new_threshold = 0.3
ypred_new_threshold = (ypred_proba[:,1] >= new_threshold).astype(int)

# Evaluate performance with new threshold
acc_new_threshold = accuracy_score(ytest, ypred_new_threshold)
confusion_new_threshold = confusion_matrix(ytest, ypred_new_threshold)
print(f"Accuracy with new threshold: {acc_new_threshold}")
print(f"Confusion matrix with new threshold:\n{confusion_new_threshold}")
print(classification_report(ytest, ypred_new_threshold))
print(f"Traing score {gbc.score(xtrain,ytrain)}")
print(f"Testing score {gbc.score(xtest,ytest)}")




# Make pickle file of our model
pickle.dump(gbc, open("model.pkl", "wb"))