## Installation Guide

Follow these steps to set up and run the project:

1. **Clone the repository and install dependencies:**  
   ```bash
   git clone https://github.com/cruvss/Class-Wise-Fact-Generator.git
   cd class-wise-fact-generator
   cd code
   pip install -r requirements.txt
   ```

2. **Create a `.env` file in the project root with your API keys:**  
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   AZURE_SPEECH_KEY=your_azure_speech_key_here
   ```

3. **Start the FastAPI server:**  
   ```bash
   uvicorn main:app --reload
   ```

4. **Launch the Streamlit interface:**  
   ```bash
   streamlit run web_app.py
   ```  
