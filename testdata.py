import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#skriver ut de første radene
print(train_data.head())
print(test_data.head())

#Forbereding av treningsdata
X_train = train_data.drop(columns=['Target', 'Id'])
y_train = train_data['Target']
X_test = test_data.drop(columns=['Id'])

#Tren model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# prediker
test_predictions = model.predict(X_test)

#Lagre prediksjon i riktig format
output = pd.DataFrame({'Id': test_data['Id'], 'Target': test_predictions})
output.to_csv('test_predictions.csv', index=False)

# Display litt av output
output.head()
