##Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

#Read dataset
data = pd.read_csv('Fashion Dataset v2.csv')
print(data.head())

# Check the data types and missing values
print(data.info())

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Extract brand and category from name
data['brand'] = data['name'].apply(lambda x: x.split(' ')[0])
data['category'] = data['products']

# Drop irrelevant columns
data.drop(['name', 'products'], axis=1, inplace=True)

# Convert categorical columns to numerical using one-hot encoding
categorical_cols = ['brand', 'category']
X = pd.get_dummies(data[categorical_cols])
y = np.log1p(data['price'])  # Log transformation for price normalization


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping])

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae:.4f}')

# Custom accuracy function
def calculate_accuracy(y_true, y_pred, threshold=0.1):
    """
    Calculate the accuracy of predictions based on a percentage threshold.

    Parameters:
    - y_true: Actual target values
    - y_pred: Predicted values
    - threshold: Percentage threshold for accuracy (default is 10%)

    Returns:
    - accuracy: Percentage of predictions within the threshold
    """
    # Calculate the absolute percentage error
    percentage_error = np.abs((y_true - y_pred) / y_true)

    # Calculate the accuracy based on the threshold
    accuracy = np.mean(percentage_error < threshold) * 100  # Convert to percentage
    return accuracy

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Reshape test_predictions to be 1-dimensional
test_predictions = test_predictions.flatten() # Reshape to 1D array

# Calculate and print the custom accuracy
accuracy = calculate_accuracy(y_test, test_predictions, threshold=0.1)  # Example threshold of 10%
print(f"Accuracy (within 10% of actual price): {accuracy:.2f}%")



# Visualize training history
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

























