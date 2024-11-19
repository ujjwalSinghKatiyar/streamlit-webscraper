import torch
from bs4 import BeautifulSoup
import requests
from torchvision import transforms
from PIL import Image
import streamlit as st

# Function to load the trained model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to scrape webpage and extract content based on tag and class
def scrape_website(url, tag, css_class):
    # Send HTTP request
    response = requests.get(url)
    
    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract elements based on tag and CSS class
    elements = soup.find_all(tag, class_=css_class)
    text_data = [element.text.strip() for element in elements]
    
    return text_data

# Function to preprocess images for prediction
def preprocess_image(image_path):
    # Define the transformation to match the model's input requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open image file
    image = Image.open(image_path)
    
    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image

# Function to make predictions using the loaded model
def predict(model, input_data):
    with torch.no_grad():
        # Get model prediction
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    
    return predicted

# Function to display results on the Streamlit app
def display_results(scraped_data, prediction):
    # Displaying the scraped data
    st.write("### Scraped Data:")
    for item in scraped_data:
        st.write(item)
    
    # Display the prediction
    st.write(f"### Prediction: {prediction.item()}")

# Streamlit user interface
def main():
    st.title("Web Scraping and Model Prediction")

    # Input fields for URL, HTML tag, and CSS class
    url = st.text_input("Enter Website URL")
    tag = st.text_input("Enter HTML tag")
    css_class = st.text_input("Enter CSS class")

    # Upload trained model
    model_file = st.file_uploader("Upload Model", type=["pt", "pth"])
    
    if model_file is not None:
        # Load the model
        model = load_model(model_file)

        # Button to start scraping and prediction
        if st.button("Scrape and Predict"):
            if url and tag and css_class:
                # Scrape data from the website
                scraped_data = scrape_website(url, tag, css_class)
                
                # Display scraped data and make predictions
                display_results(scraped_data, model)
            else:
                st.warning("Please provide all inputs.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
