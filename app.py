import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details
    
# Prediction function
def predict(img, interpreter, input_details, output_details, class_names):
    img = img.resize((224, 224))  # same size used in training
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = np.argmax(output)
    confidence = float(output[pred_idx])
    return class_names[pred_idx], confidence, output

# Streamlit UI
st.title("🧍 Human Action Recognition")
st.write("Upload an image to classify the human action.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")
    interpreter, input_details, output_details = load_model()

    # Replace with your actual class names from training
    class_names = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping', 'texting', 'using_laptop']  

    label, confidence, probs = predict(image, interpreter, input_details, output_details, class_names)

    st.success(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")

    # Show probability chart
    st.bar_chart({class_names[i]: probs[i] for i in range(len(class_names))})



