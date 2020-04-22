# librairie pour les opérations de base
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# librairie pour la visualisation des données
import matplotlib.pyplot as plt
import seaborn as sns

# librairie pour le Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Step 1 Open CSV file

# Creation of a dataset with the csv file
dfCsv = pd.read_csv('C:\Users\Mikazuki\Desktop\TP_Wines\winemag-data-130k-v2.csv', sep=",")

# Step 2 Copy of dataset
df = dfCsv.copy()

df
# See images/df.PNG


# Step 3 Own dataset creation with (points, price)
dfC = df[['points', 'price']]
dfC
# See images/dfC.PNG

# Creation of another dataset with (points, price, country)
dfG = df[['points', 'price','country']]
dfG
# See images/dfG.PNG


# Step 4 Grouping of points by country with aggregation average and standard deviation
dfG.groupby('country').points.agg(['mean', 'std'])
# See images/dfGG.PNG


# To use the matplotlib plot on the line
%matplotlib inline

# Small visualization of the result
dfG.groupby('country').points.agg(['mean', 'std']).plot(kind='bar')
# See images/dfGV.PNG

# Bonus 1 We select the 5 values which have a price <10 and points> 88
dfG[(dfG.price < 10) & (dfG.points > 88)].head(5)
# See images/dfGB1.PNG

# Bonus 1 We select the 5 values which have a price <30 and points> 90 and which are from Chile
dfG[(dfG.price < 30) & (dfG.points > 90) & (dfG.country == "Chile")].head(5)
# See images/dfGB12.PNG

%matplotlib inline

# Bonus 2 Visualization of the points compared to the price
X = dfG[['points']]
y = dfG[['price']]

# We draw the graph with the points on the abscissa and the prices on the ordinate
sns.pairplot(dfG, x_vars=['points'], y_vars='price', size=6)
# See images/dfGv1.PNG

# View points by country
# We draw the graph with the points on the abscissa and the country on the ordinate
sns.pairplot(dfG, x_vars=['points'], y_vars='country', height=6)
# See images/dfGv2.PNG

# Visualization of the points compared to the price
# We draw the graph with the points on the abscissa and the prices on the ordinate
sns.pairplot(dfG[["points", "price"]], diag_kind="kde")
# See images/dfgv3.PNG

dfG
# See images/dfGM.PNG


# Bonus 3
# Preprocessing 
# we browse the columns all those that are numeric go to columns
columns = []

for column in dfG.columns:
    if (dfG[column].dtype == 'int64') | (dfG[column].dtype == 'float64'):
        columns.append(dfG[column].name)

print("Task completed")

# Here It is to do Data Standardization

# scaler = StandardScaler() # scaler -1 / 1
# dfG[columns] = scaler.fit_transform(dfG[columns])

# Standardization of the numerical data which are in columns
MMS = MinMaxScaler()        # scaler 0 / 1
dfG[columns] = MMS.fit_transform(dfG[columns])

print("Task completed")

# We will demmifier the values of the country column to have a column by country
df_categorical = df[['country']]
df_dummies = pd.get_dummies(df_categorical)

# We gather our tables to create the final dataset and we remove the contry column
df_final = pd.concat([df_dummies, dfG], axis=1).drop(
    columns=df_categorical.columns, axis=1)

df_final
# See images/dfGM1.PNG

# Remove NaN values
simpleImputer = SimpleImputer(missing_values=np.nan)

df_final[['points']] = simpleImputer.fit_transform(df_final[['points']])
df_final[['price']] = simpleImputer.fit_transform(df_final[['price']])

# the data (features) go into the X and the variable that we want to predict in the y (Target)
X = df_final.drop(['points'], axis=1)
y = df_final[['points']]

# Separation between the trainset (X_train, y_train) for training the model and the testtset (X_test, y_test) to evaluate and predict the values of the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Model régréssion LinearRegression
linereg = LinearRegression()

# Model training on the trainset
linereg.fit(X_train, y_train)

# Predictions of sclaler points
linereg.predict(X_test)
# See images/linePred.PNG

# Score from our LinearRegression model
linereg.score(X_test, y_test)
# See images/lineScore.PNG

# The RandomForestRegressor regression model
rfr = RandomForestRegressor(n_estimators=100, max_depth=3)

# Model training on the trainset
rfr.fit(X_train, y_train)

# Predictions of sclaler points
rfr.predict(X_test)
# See images/rfrPred.PNG

# Score from our RandomForestRegressor model
rfr.score(X_train, y_train)
# See images/rfrScore.PNG

# GridSearchCV to Train the model to find the best parameters
param_grid = {'n_estimators': [50, 100, 150, 200],
              'max_depth': [3, 6, 10],
              }
# Créer un RandomForestRegressor avec recherche de hyperparamètre par validation croisée
rf = GridSearchCV(
    estimator=RandomForestRegressor(),                  # RandomForestRegressor rfr
    param_grid=param_grid,                              # hyperparamètres à tester
    # nombre de folds de validation croisée
    cv=5,
    verbose=0
)
# Model training on the trainset
rf.fit(X_train, y_train)

# Returns the best parameters for the model
rf.best_params_
