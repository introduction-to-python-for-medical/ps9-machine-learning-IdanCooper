
import pandas as pd

DataFrame1 = pd.read_csv('parkinsons.csv')
DataFrame = DataFrame1.dropna()


features = ['PPE' , 'DFA']
target = 'status'
x = DataFrame[features]
y = DataFrame[target]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_scaled

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42) 

from sklearn.svm import SVC

model = SVC()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
