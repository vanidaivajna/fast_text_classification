import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# Step 1: Load and preprocess the data
data = pd.read_csv("data.csv")  # Replace with the path to your dataset
texts = data['text'].tolist()
labels = data['intent'].tolist()

# Step 2: Split the data into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Step 3: Tokenize the texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Step 4: Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

# Step 5: Define the BERT-based model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Step 6: Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=model.compute_loss,
              metrics=['accuracy'])

model.fit(train_dataset.shuffle(1000).batch(16),
          epochs=3,
          batch_size=16,
          validation_data=val_dataset.shuffle(1000).batch(16))

# Step 7: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset.batch(16), verbose=2)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
