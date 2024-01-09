import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

normal_df = pd.read_csv('../MLOPS-Pipeline/youtube data.csv', low_memory=False)
abnormal_df = pd.read_csv('../MLOPS-Pipeline/malicious.csv', low_memory=False)
# drop the column malicious
abnormal_df = abnormal_df.drop('malicious', axis=1)
normal_df = normal_df.drop('malicious', axis=1)

# remove the column pid from both dataframes
abnormal_df = abnormal_df.drop('pid', axis=1)
normal_df = normal_df.drop('pid', axis=1)

normal_df['label'] = 0
abnormal_df['label'] = 1

print("Normal DF is = ", normal_df.shape)


# # rename malicious to label
# abnormal_df = abnormal_df.rename(columns={'malicious': 'label'})
# normal_df = normal_df.rename(columns={'normal': 'label'})


# now mix both dataframes
total_df = pd.concat([normal_df, abnormal_df], ignore_index=True)

# drop first column
total_df = total_df.drop(total_df.columns[0], axis=1)

# shuffle the data
total_df = total_df.sample(frac=1).reset_index(drop=True)

# strip the data so that no space is left
total_df = total_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print(total_df.dtypes)

print(total_df.shape)
# drop all the null columns 
total_df = total_df.dropna(axis=1, how='all')

# remove all columns whose sum is zero
total_df = total_df.loc[:, (total_df != 0).any(axis=0)]
total_df = total_df.dropna(subset=['label'])
total_df['label'] = total_df['label'].astype(int)

# Assuming 'df' is your pandas DataFrame containing the dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Extract features and target variable
X = total_df[['Cpu','memory', 'Network']].values
y = total_df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize/Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the input data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.6)


# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# take a random row from the dataframe
row = normal_df.iloc[0]
print(row)
cpu = row['Cpu']
memory = row['memory']
Network = row['Network']

orig = row['label']

new_data = np.array([[cpu, memory, Network]])

new_data = np.array([[cpu, memory, Network]])  # Replace '...' with your actual feature values
new_data_scaled = scaler.transform(new_data)
new_data_reshaped = new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])
predictions = model.predict(new_data_reshaped)
print (predictions)

# # save the model
# model.save('testing.h5')
# # save the scaler
# import joblib
# joblib.dump(scaler, 'scaler1.pkl')

# print the confusion matrix
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# print fasle positive rate and true positive rate
print("False positive rate is = ", fp / (fp + tn))
print("True positive rate is = ", tp / (tp + fn))

# save model 
model.save('testing.h5')
