from copyreg import pickle
from gravityai import gravityai as grav # Import the gravityai library 
import pickle as pkl # Import the pickle library for saving and loading data
import pandas as pd # Import the pandas library

model = pickle.load(open('financial_text_classifier.pkl', 'rb')) # Load the model from the pickle file 
tfidf_vectorizer = pkl.load(open('financial_text_vectorizer.pkl', 'rb')) # Load the tfidf vectorizer from the pickle file
label_encoder = pkl.load(open('financial_text_encoder.pkl', 'rb')) # Load the label encoder from the pickle file

def process(inPath, outPath):
    input_df = pd.read_csv(inPath) # Read the input csv file
    
    features = tfidf_vectorizer.transform(input_df['text']) # Transform the text column into a tfidf vector (Vectorize the data)
    
    predictions = model.predict(features) # Predict the labels for the data using the model
    
    input_df['category'] = label_encoder.inverse_transform(predictions) # Transform the predicted labels into the original labels using the label encoder 
    
    output_df = input_df[['id', 'category']] # Create a new dataframe with the id and category columns 
    output_df.to_csv(outPath, index=False) # Write the output dataframe to a csv file
    
grav.wait_for_requests(process) # Wait for a request from the client and then process the data