# Import necessary libraries
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load MNIST dataset from OpenML (you can also use sklearn's inbuilt datasets)
mnist = fetch_openml('mnist_784', version=1)

# Data preprocessing
X = mnist['data'] / 255.0  # Normalize the data (pixels between 0 and 1)
y = mnist['target'].astype(int)  # Labels (digits 0-9)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")

# Display the first 5 test images and their predicted labels
for i in range(5):  # You can change the range to display more images (e.g., 10 or more)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
    plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns

# def introduction_to_ml():
#     """
#     A simple introduction to Machine Learning with Neural Networks
#     Using the MNIST Digit Recognition Dataset
#     """
#     # 1. Load the Digits Dataset
#     print("=== Step 1: Loading Digit Dataset ===")
#     digits = load_digits()
#     X, y = digits.data, digits.target
    
#     # Print basic dataset information
#     print(f"Total number of samples: {len(X)}")
#     print(f"Image shape: {digits.images[0].shape}")
#     print(f"Number of classes: {len(np.unique(y))}")
    
#     # 2. Visualize Some Sample Digits
#     print("\n=== Step 2: Visualizing Sample Digits ===")
#     plt.figure(figsize=(10, 5))
#     for i in range(10):
#         plt.subplot(2, 5, i+1)
#         plt.imshow(digits.images[i], cmap='gray')
#         plt.title(f'Digit: {digits.target[i]}')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('sample_digits.png')
#     plt.close()
    
#     # 3. Prepare the Data
#     print("\n=== Step 3: Preparing Data ===")
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # 4. Create and Train Neural Network
#     print("\n=== Step 4: Training Neural Network ===")
#     # Create a Multi-Layer Perceptron Classifier
#     mlp = MLPClassifier(
#         hidden_layer_sizes=(100, 50),  # Two hidden layers
#         max_iter=500,
#         activation='relu',
#         solver='adam',
#         random_state=42
#     )
    
#     # Train the model
#     mlp.fit(X_train_scaled, y_train)
    
#     # 5. Evaluate the Model
#     print("\n=== Step 5: Model Evaluation ===")
#     # Make predictions
#     y_pred = mlp.predict(X_test_scaled)
    
#     # Print classification report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
    
#     # Create Confusion Matrix Visualization
#     plt.figure(figsize=(10, 8))
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
#     plt.close()
    
#     # 6. Visualize Predictions
#     print("\n=== Step 6: Visualizing Predictions ===")
#     plt.figure(figsize=(12, 6))
#     for i in range(10):
#         plt.subplot(2, 5, i+1)
#         plt.imshow(X_test.reshape(-1, 8, 8)[i], cmap='gray')
#         plt.title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('predictions.png')
#     plt.close()

#     # Print overall accuracy
#     accuracy = mlp.score(X_test_scaled, y_test)
#     print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# def main():
#     print("Machine Learning Project: Digit Recognition")
#     introduction_to_ml()

# if __name__ == "__main__":
#     main()