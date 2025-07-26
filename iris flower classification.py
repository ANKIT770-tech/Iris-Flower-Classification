import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Example model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data  # Features
y = iris.target # Target (species)
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
# Map numerical target to actual species names for better readability
df['species_name'] = df['species'].apply(lambda x: target_names[x])

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nSpecies Distribution:")
print(df['species_name'].value_counts())


sns.pairplot(df, hue='species_name', palette='viridis')
plt.suptitle('Pair Plot of Iris Features by Species', y=1.02) # Adjust suptitle position
plt.show()

# d. Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# e. Model Selection (Using K-Nearest Neighbors as an example)
model = KNeighborsClassifier(n_neighbors=3) # You can experiment with n_neighbors

# f. Model Training
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# g. Model Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))


# h. Example of making a prediction on new, unseen data
# if we have a new flower with these measurements:
# sepal length=5.5, sepal width=2.5, petal length=4.0, petal width=1.3
new_flower_measurements = [[5.5, 2.5, 4.0, 1.3]]
predicted_species_index = model.predict(new_flower_measurements)
predicted_species_name = target_names[predicted_species_index[0]]

print(f"\nNew flower measurements: {new_flower_measurements[0]}")
print(f"Predicted species: {predicted_species_name}")
