import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("house_price.csv")
print(df.head())
df.fillna(df.mean(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)

X= df.drop("price", axis=1)
y =df["price"]

X_train, X_test, y_train, y_test =train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


sample_house = X.iloc[0:1]
predicted_price =  model.predict(sample_house)
print("Predicted House Price: ", predicted_price[0])