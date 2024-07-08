df_in = SAS.sd2df(_input1)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


print(f" the data set length is: {len(df_in)}")

# EDA - histograms of columns
for column in df_in.select_dtypes(exclude='object'):
	plt.figure(figsize=(10,8))
	plt.hist(df_in[column],bins=30,edgecolor='k', alpha=0.7)
	plt.title(f'distribution of {column}')
	plt.xlabel(column)
	plt.ylabel('freq')
	plt.grid(True)
	plt.show()


for column in df_in.isnull().sum()[df_in.isnull().sum()>1].index:
    df_in[column].fillna(df_in[column].mean(), inplace =True)



# remove records from q.99 and above
q = df_in['person_emp_length'].quantile(0.99)
df_cleaned = df_in[df_in['person_emp_length']<q]

q = df_in['person_income'].quantile(0.99)
df_cleaned = df_in[df_in['person_income']<q]


# Encoding
lencoder = LabelEncoder()
for column in df_cleaned.select_dtypes(include='object'):
    df_cleaned[column] = lencoder.fit_transform(df_cleaned[column])


# Train, Test

X = df_cleaned.drop(columns = ['loan_status'])
Y = df_cleaned['loan_status']

from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()

X_scaled = stdscaler.fit_transform(X)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size =0.2,random_state=42)

print(f"x train shape: {x_train.shape}")
print(f"x test shape: {x_test.shape}")
print(f"y train shape: {y_train.shape}")
print(f"y test shape: {y_test.shape}")


# moedling
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)


# Evaluation
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy}")
y_pred_df = pd.DataFrame(y_pred,columns=['Predicted'])
# back to sas data

SAS.df2sd(y_pred_df,_output1)