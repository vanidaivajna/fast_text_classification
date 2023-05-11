import random
from sklearn.model_selection import train_test_split

# Step 1: Load labeled data
data = [
    ("This is a sample text 1.", "intent1"),
    ("Another text for intent1.", "intent1"),
    ("Some text for intent2.", "intent2"),
    ("This is a sample text 2.", "intent2"),
    ("A different text for intent3.", "intent3"),
    ("Text related to intent3.", "intent3")
    # Add more data samples...
]

# Step 2: Separate data by intent labels
intent_data = {}
for text, intent in data:
    if intent not in intent_data:
        intent_data[intent] = []
    intent_data[intent].append(text)

# Step 3: Shuffle samples within each intent category
for intent in intent_data:
    random.shuffle(intent_data[intent])

# Step 4: Prepare data for train-test split
texts = []
labels = []
for intent, samples in intent_data.items():
    texts.extend(samples)
    labels.extend([intent] * len(samples))

# Step 5: Split data into train, validation, and test sets (with stratification)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)
train_texts, valid_texts, train_labels, valid_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# Step 6: Print the resulting datasets
print("Train set:")
for text, label in zip(train_texts, train_labels):
    print(label, text)
print("\nValidation set:")
for text, label in zip(valid_texts, valid_labels):
    print(label, text)
print("\nTest set:")
for text, label in zip(test_texts, test_labels):
    print(label, text)
