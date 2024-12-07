import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load data
data = pd.read_csv("data.csv", sep=",", encoding="utf-8")
data['datetime'] = pd.to_datetime(data['datetime'])

# Features and target
features = ["tempmax", "tempmin","feelslikemax","feelslikemin", "humidity", "windspeed", "precip"]
target = "temp"

# Separate train and test data
test_data = data.tail(33)
train_data = data.iloc[:-33]

# Clean data
train_data.dropna(subset=features + [target], inplace=True)
test_data.dropna(subset=features + [target], inplace=True)

# Store original min and max for scaling
temp_min = train_data[target].min()
temp_max = train_data[target].max()

# Separate scalers for features and target
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(temp_min, temp_max))

# Scale features
X_train_features = feature_scaler.fit_transform(train_data[features])
X_test_features = feature_scaler.transform(test_data[features])

# Flatten the target column to 1D before scaling
y_train = target_scaler.fit_transform(train_data[target].values.reshape(-1, 1)).flatten()
y_test = target_scaler.transform(test_data[target].values.reshape(-1, 1)).flatten()

# Sequence creation function
def create_sequences(features, target, sequence_length=3):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

# Prepare sequences
X_train, y_train = create_sequences(X_train_features, y_train)
X_test, y_test = create_sequences(X_test_features, y_test)

# Enhanced model architecture
model = Sequential([
    GRU(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.3),
    GRU(32, activation='tanh'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

# Advanced training callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Compile with custom learning rate
model.compile(optimizer=Adam(learning_rate=0.0016), loss='mse', metrics=['mae'])

# Training
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    verbose=1, 
    callbacks=[early_stopping, reduce_lr]
)

# Calculate accuracy for training data
train_predictions = model.predict(X_train)
train_predictions = target_scaler.inverse_transform(train_predictions.reshape(-1, 1))
y_train_original = target_scaler.inverse_transform(y_train.reshape(-1, 1))
train_mae = np.mean(np.abs(train_predictions - y_train_original))
train_accuracy = 1 - (train_mae / np.mean(np.abs(y_train_original)))

# Calculate accuracy for test data
test_predictions = model.predict(X_test)
test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1, 1))
y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
test_mae = np.mean(np.abs(test_predictions - y_test_original))
test_accuracy = 1 - (test_mae / np.mean(np.abs(y_test_original)))

# Print accuracy percentages
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Visualization of training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualization of predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label="Actual Temperature")
plt.plot(test_predictions, label="Predicted Temperature")
plt.title("Temperature Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# Performance metrics
mae = np.mean(np.abs(test_predictions - y_test_original))
mse = np.mean((y_test_original - test_predictions) ** 2)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Daily error table for November 2024
days_in_november = pd.date_range(start="2024-11-01", end="2024-11-30")
errors = np.abs(y_test - test_predictions.flatten())

# Hata tablosu hazırlama
error_table = pd.DataFrame({
    "Date": days_in_november[:len(errors)], 
    "Error": errors
})

print(error_table)


error_matrix = np.zeros((5, 7))  # 5 hafta x 7 gün

#First day of November is Friday
start_day = 4  

# Ensure error_table has enough rows
error_table = error_table.head(error_matrix.size)

# Create a flattened view of error matrix
error_matrix_flat = error_matrix.ravel()

# Start from the first available column of the first week
start_col = start_day  # 4 for Friday

# Fill the matrix
for i in range(len(error_table)):
    week = (i + start_col) // 7
    day_of_week = (i + start_col) % 7
    
    if week < error_matrix.shape[0]:
        error_matrix[week, day_of_week] = error_table['Error'].iloc[i]

# Görselleştirme
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(error_matrix, cmap="YlGnBu", interpolation="nearest")

# X ve Y eksenine gün ve hafta isimlerini ekleyin
ax.set_xticks(np.arange(7))
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_yticks(np.arange(5))
ax.set_yticklabels([f"Week {i+1}" for i in range(5)])

# Her hücreye hata değerini yazdır
for i in range(5):
    for j in range(7):
        # 0 olan hücrelere değer yazdırılmasın
        text = ax.text(j, i, f"{error_matrix[i, j]:.2f}" if error_matrix[i, j] != 0 else "",
                       ha="center", va="center", color="black")

# Başlık ve renk barı
ax.set_title("November 2024 Daily Prediction Errors")
fig.colorbar(im, ax=ax, orientation='vertical', label="Error (°C)")
plt.show()
