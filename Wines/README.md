# TP_Wines
# Step 1 Open CSV file
We import the csv to convert it into a dataset


# Step 2 Copy of dataset
We copy the dataset so as not to touch the original


# Step 3 Own dataset creation with (points, price)
We create a new dataset which will contain the points and price columns which come from the df dataset


# Step 4 Grouping of points by country with aggregation average and standard deviation
We create another dataset which will contain the points, the prices and the countries then we will make a grouping to see the average and the standard deviation of the points according to a country

We then have a visualization with the country graph

# Bonus 1
We recover the first 5 values which have a price below 10 and points above 88

We recover the first 5 values which have a price below 30 and points above 90 and which are from Chile

# Bonus 2
We visualize the points according to the price and also the points according to the country

# Bonus 3
The prediction of points for a wine with the country and the price of entry.
3 steps:
- Very important preprocessing
- Model training (LinearRegression and RandomForestRegressor)
- GridSearchCV to find the optimal parameters for the model and have a better score and thus better predictions