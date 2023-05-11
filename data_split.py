import random
from sklearn.model_selection import train_test_split
import pandas as pd

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

# Step 6: Create new DataFrames for train, validation, and test sets
train_data = pd.DataFrame({"text": train_texts, "intent": train_labels})
valid_data = pd.DataFrame({"text": valid_texts, "intent": valid_labels})
test_data = pd.DataFrame({"text": test_texts, "intent": test_labels})

# Step 7: Print the resulting datasets
print("Train set:")
print(train_data)
print("\nValidation set:")
print(valid_data)
print("\nTest set:")
print(test_data)
