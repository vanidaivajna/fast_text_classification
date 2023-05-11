import fasttext
import random
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

# Step 1: Load labeled data into a Pandas DataFrame
data = pd.DataFrame([
    {"text": "This is a sample text 1.", "intent": "intent1"},
    {"text": "Another text for intent1.", "intent": "intent1"},
    {"text": "Some text for intent2.", "intent": "intent2"},
    {"text": "This is a sample text 2.", "intent": "intent2"},
    {"text": "A different text for intent3.", "intent": "intent3"},
    {"text": "Text related to intent3.", "intent": "intent3"}
    # Add more data samples...
])

# Step 2: Separate data by intent labels
intent_data = {}
for _, row in data.iterrows():
    text, intent = row["text"], row["intent"]
    if intent not in intent_data:
        intent_data[intent] = []
    intent_data[intent].append(text)

# Step 3: Shuffle samples within each intent category
for intent in intent_data:
    random.shuffle(intent_data[intent])

# Step 4: Prepare data for cross-validation
texts = []
labels = []
for intent, samples in intent_data.items():
    texts.extend(samples)
    labels.extend([intent] * len(samples))

# Step 5: Perform cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []
for fold, (train_indices, val_indices) in enumerate(kfold.split(texts), 1):
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    # Step 6: Create a training data file
    train_data_file = "train_data.txt"
    with open(train_data_file, "w") as f:
        for text, label in zip(train_texts, train_labels):
            line = f"__label__{label} {text}\n"
            f.write(line)

    # Step 7: Train a FastText supervised model
    model = fasttext.train_supervised(input=train_data_file)

    # Step 8: Evaluate the model on the validation set
    predictions = model.predict(val_texts)
    predicted_labels = [label[0].replace("__label__", "") for label in predictions[0]]
    
    # Calculate metrics for the fold
    fold_report = classification_report(val_labels, predicted_labels, output_dict=True)
    fold_metrics.append(fold_report)

    # Print metrics for the fold
    print(f"Fold {fold} Metrics:")
    print(classification_report(val_labels, predicted_labels))

# Step 9: Calculate average metrics across folds
overall_metrics = {}
class_metrics = {}
for metric_name in fold_metrics[0]:
    metric_values = [fold[metric_name] for fold in fold_metrics]
    if metric_name != "accuracy":
        overall_metrics[metric_name] = sum(metric_values) / len(metric_values)
        class_metrics[metric_name] = {intent: metric for intent, metric in zip(intent_data.keys(), metric_values)}

# Print overall metrics
print("Overall Metrics:")
print(overall_metrics)

# Print class-level metrics
print("Class-level Metrics:")
print(class_metrics)
