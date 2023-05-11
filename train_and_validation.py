import fasttext
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Convert train data to FastText format
train_data_file = "train_data.txt"
train_data = train_data[['intent', 'text']]  # Assuming your train data DataFrame has 'intent' and 'text' columns
train_data['label'] = '__label__' + train_data['intent']
train_data[['label', 'text']].to_csv(train_data_file, sep='\t', index=False, header=False)

# Step 2: Train a FastText supervised model
model = fasttext.train_supervised(input=train_data_file)

# Step 3: Prepare validation data
val_texts = val_data['text'].tolist()  # Assuming 'text' column in your validation data DataFrame
val_labels = val_data['intent'].tolist()  # Assuming 'intent' column in your validation data DataFrame

# Step 4: Predict labels on the validation data
predictions = model.predict(val_texts)
predicted_labels = [label[0].replace("__label__", "") for label in predictions[0]]

# Step 5: Generate classification report
report = classification_report(val_labels, predicted_labels)
print("Classification Report:")
print(report)

# Step 6: Generate confusion matrix
cm = confusion_matrix(val_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=model.labels, yticklabels=model.labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
