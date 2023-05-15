import pandas as pd
import fasttext
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Step 1: Divide the data into train and validation sets
train_data, val_data = train_test_split(data, train_size=0.8, random_state=42)

# Step 2: Convert train data to FastText format
train_file = "train_data.txt"
train_data[['intent', 'text']].to_csv(train_file, sep='\t', index=False, header=False)

# Step 3: Define a function to evaluate the model on F1 score
def evaluate_model(model, validation_data):
    val_texts = validation_data['text'].tolist()  # Assuming 'text' column in your validation data DataFrame
    val_labels = validation_data['intent'].tolist()  # Assuming 'intent' column in your validation data DataFrame

    # Step 4: Predict labels on the validation data
    predictions = model.predict(val_texts)
    predicted_labels = [label[0].replace("__label__", "") for label in predictions[0]]

    # Calculate F1 score
    f1 = f1_score(val_labels, predicted_labels, average='weighted')

    return f1

# Step 5: Hyperparameter tuning
best_f1 = 0
best_model = None
num_trials = 10  # Number of random trials

for trial in range(num_trials):
    # Generate random hyperparameters
    lr = random.uniform(0.1, 1.0)  # Learning rate between 0.1 and 1.0
    epoch = random.randint(5, 50)  # Number of training epochs between 5 and 50

    # Train the FastText model
    model = fasttext.train_supervised(
        input=train_file,
        lr=lr,
        epoch=epoch,
        loss='softmax'
    )

    # Evaluate the model on validation data
    f1 = evaluate_model(model, val_data)

    # Update the best model and F1 score if the current F1 score is better
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

    print(f"Trial {trial + 1}: F1 Score = {f1}, Learning Rate = {lr}, Epochs = {epoch}")

# Step 6: Use the best model for testing or inference
# Evaluate the best model on test data or use it for inference
