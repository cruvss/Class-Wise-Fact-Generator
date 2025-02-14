import streamlit as st
import requests

# Define FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000/generate-facts"

# Streamlit UI
st.title("Class-Wise Fact Generator")
st.write("Enter a topic and grade level to generate facts.")

# User inputs
topic = st.text_input("Enter a topic:", "Newton")
grade_level = st.number_input("Enter grade level:", min_value=1, max_value=12, value=5, step=1)

if st.button("Generate Facts"):
    # Send request to FastAPI
    payload = {"topic": topic, "grade_level": grade_level} # make sure the payload mathces the pydantic model of the FastAPI
    response = requests.post(FASTAPI_URL, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        st.subheader("Generated Facts:")
        
        # Process and display facts
        for fact in data["facts"]:
            st.write(fact)
        
        # Display audio player
        st.subheader("Generated Audio:")
        audio_urls = data["audio_urls"]  
        st.write(f"Displaying the First Audio URL: {audio_urls[0]}")  
        st.audio(audio_urls[0], format="audio/wav") 
    
    else:
        st.error("Failed to generate facts. Please check the FastAPI server.")
