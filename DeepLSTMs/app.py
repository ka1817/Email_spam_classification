import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your saved model and tokenizer
def load_model_and_tokenizer():
    # Assuming the model is saved as 'spam_ham_model.h5' and tokenizer saved as 'tokenizer.pickle'
    model = tf.keras.models.load_model('model.h5')

    # You need to have a way to load the tokenizer that you used
    import pickle
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer

# Preprocessing function for the user input
def preprocess_input(texts, tokenizer, maxlen=50):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, padding='post')

# Prediction function
def predict_text(model, tokenizer, sample_texts, maxlen=50):
    X_predict = preprocess_input(sample_texts, tokenizer, maxlen)
    predictions = model.predict(X_predict)
    
    results = []
    for text, pred in zip(sample_texts, predictions):
        label = "spam" if pred[0] > 0.5 else "ham"
        results.append({
            "Text": text,
            "Predicted Label": label,
            "Prediction Confidence": f"{pred[0]:.4f}"
        })
    return results

# Streamlit App Interface
def main():
    st.title('Spam vs Ham Text Classifier')
    st.markdown("""
    This is a simple Streamlit app that predicts whether a given text is **Spam** or **Ham** using a pre-trained model.
    """)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Text input
    text_input = st.text_area("Enter the text you want to classify:")

    # Button to predict
    if st.button("Predict"):
        if text_input:
            # Get the prediction
            prediction_results = predict_text(model, tokenizer, [text_input])
            
            # Display the result
            for result in prediction_results:
                st.write(f"**Text**: {result['Text']}")
                st.write(f"**Predicted Label**: {result['Predicted Label']}")
                st.write(f"**Prediction Confidence**: {result['Prediction Confidence']}")
        else:
            st.error("Please enter some text to classify.")

# Run the app
if __name__ == "__main__":
    main()
