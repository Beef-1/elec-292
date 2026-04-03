import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

iter_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
train_accs = []
test_accs = []

for it in iter_values:
    model = LogisticRegression(max_iter=it)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accs.append(accuracy_score(y_train, y_train_pred))
    test_accs.append(accuracy_score(y_test, y_test_pred))

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)

print("Final training accuracy:", final_model.score(X_train, y_train))
print("Final test accuracy:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

plt.plot(iter_values, train_accs, marker='o', label="Training Accuracy")
plt.plot(iter_values, test_accs, marker='o', label="Test Accuracy")
plt.xlabel("max_iter")
plt.ylabel("Accuracy")
plt.title("Training Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()