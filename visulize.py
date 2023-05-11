import fasttext
import ipywidgets as widgets
from IPython.display import display

# Load the trained FastText model
model = fasttext.load_model("your_model.bin")  # Replace "your_model.bin" with the path to your trained model file

# Create UI elements
text_input = widgets.Textarea(description="Enter text:")
button = widgets.Button(description="Predict")
output_label = widgets.Label()

# Define callback function for button click event
def predict_labels(button):
    text = text_input.value
    # Predict label using the trained model
    label = model.predict(text)[0][0].replace("__label__", "")
    output_label.value = "Predicted Label: " + label

# Register callback function for button click event
button.on_click(predict_labels)

# Display UI
display(text_input, button, output_label)
