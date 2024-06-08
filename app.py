import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load x_tokenizer
with open('x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)

# Load y_tokenizer
with open('y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

# Load the saved model
model = load_model('transformer_model.h5')

max_eng_len = 10
max_fra_len = 19 

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask
    
def translate(sentence):
    # Tokenize and encode the input sentence
    inputs = x_tokenizer.texts_to_sequences([sentence])
    inputs = pad_sequences(inputs, maxlen=max_eng_len, padding='post')
    
    # Initialize the decoder input with the special token 'start_'
    decoder_input = tf.expand_dims([y_tokenizer.word_index['start_']], 0)
    
    # Initialize the output sentence
    output_sentence = []
    
    # Masks for inference
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, decoder_input)
    
    # Run inference
    for i in range(max_fra_len):
        predictions = model(inputs, decoder_input, False, enc_padding_mask, combined_mask, dec_padding_mask)
        
        # Get the predicted token ID
        predicted_id = tf.argmax(predictions, axis=-1)[0, -1].numpy()
        
        # If the predicted token is 'end_', stop inference
        if y_tokenizer.index_word[predicted_id] == '_end':
            break
        
        # Append the predicted token to the output sentence
        output_sentence.append(y_tokenizer.index_word[predicted_id])
        
        # Update the decoder input for the next timestep
        decoder_input = tf.concat([decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1)
        
        # Update masks for the next timestep
        _, combined_mask, dec_padding_mask = create_masks(inputs, decoder_input)
    
    # Join the tokens to form the output sentence
    return ' '.join(output_sentence)

# Streamlit App
st.title("English to French Translator")

# Input text box for user input
input_sentence = st.text_input("Enter a sentence in English:")

# Translate button
if st.button("Translate"):
    if input_sentence:
        output_sentence = translate(input_sentence)
        st.write("Translated Sentence:", output_sentence)
    else:
        st.write("Please enter a sentence.")
