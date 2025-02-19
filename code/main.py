
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from typing import Optional, Dict
from openai import OpenAI
import httpx

from openai import OpenAI # this is for generating facts
import azure.cognitiveservices.speech as speechsdk  #for text to speech conversion tasks

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
# above code is for logging purpose it will log all the information in app.log file,

logger = logging.getLogger(__name__)

FastAPI_URL = "http://127.0.0.1:8000"

# Initialize FastAPI app
app = FastAPI(
    title="Class-Wise Fact Generator API",
    description="API for generating class-specific facts and converting them to audio"
)

# Azure Speech Configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SERVICE_REGION = "eastus"  
speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)

# OpenAI Configuration
OPENAI_API_KEY = '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' #os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

# Below we have defined the Pydantic mode for the request and response of the API 
# For fact request we have defined the topic and grade level
# for fact response we have defined the facts, grade level, timestamp and audio_url
# for audio request we have defined the text and tone
# for audio response we have defined the audio_url and timestamp

class FactRequest(BaseModel):
    topic: str
    grade_level: int
    language : str 
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Newton",
                "grade_level": 5,
                "language": "English" # now we have added language field in the request itself
            }
        }

class FactResponse(BaseModel):
    facts: List[str]
    grade_level: int
    timestamp: str
    audio_urls: List[str]  # Changed from audio_url to audio_urls
    tones: List[str]  # also tones is added in the response for you to visuzliase 
    
class AudioRequest(BaseModel):
    text: List[str]
    tone: List[str]
    language : str  # this is added in the request itself
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": ["Newton discovered gravity."],
                "tone": ["happy"],
                "language": "Hindi"
            }
        }

class AudioResponse(BaseModel):
    audio_url: List[str]
    timestamp: str

class ResponseGenerator:
    def __init__(
        self, 
        api_key: str, 
        base_url: str = OPENAI_BASE_URL,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Language-specific templates
        self._language_templates = {
            "english": {
                "system_msg": """You are an expert educator who provides structured facts in English tailored to students of different grade levels. 
                Each fact should be engaging, structured, and appropriate for the given class level.
                Additionally, define the tone of narration explicitly for each fact so that it can be used for Text-to-Speech (TTS) conversion. 
                Please restrict tones to 'cheerful','narration-professional', 'friendly' or 'chat'."""
            },
            "hindi": {
                "system_msg": """आप एक विशेषज्ञ शिक्षक हैं जो विभिन्न कक्षा स्तरों के छात्रों के लिए हिंदी में संरचित तथ्य प्रदान करते हैं।
                प्रत्येक तथ्य आकर्षक, संरचित और दी गई कक्षा के स्तर के लिए उपयुक्त होना चाहिए।
                साथ ही, प्रत्येक तथ्य के लिए कथन का स्वर स्पष्ट रूप से परिभाषित करें।
                कृपया स्वर को 'cheerful','narration-professional', 'friendly', या 'chat' तक सीमित रखें।"""
            }
        }

    def _get_example_by_grade(self, grade: int, language: str) -> str:
        """Get appropriate example based on grade level and language."""
        lang = language.lower()
        
        if lang == "hindi":
            if grade < 5:
                return """
                {
                    "topic": "न्यूटन",
                    "grade_level": "प्राथमिक विद्यालय",
                    "facts": [
                        {
                            "fact": "क्या आप जानते हैं कि आइजैक न्यूटन बिल्कुल आपकी तरह थे - दुनिया के बारे में बहुत जिज्ञासु? जब वे बच्चे थे, वे हमेशा सवाल पूछते थे और जानना चाहते थे कि चीजें कैसे काम करती हैं। उनकी जिज्ञासा ने उन्हें विज्ञान में कुछ सबसे बड़ी खोजें करने में मदद की।",
                            "tone": "cheerful"
                        },
                        {
                            "fact": "क्या आपने कभी पेड़ से सेब गिरते देखा है? एक दिन, न्यूटन ने भी ऐसा ही देखा! उन्होंने सोचा - चीजें हमेशा नीचे क्यों गिरती हैं? वे आसमान में क्यों नहीं उड़ती? इस सवाल ने उन्हें गुरुत्वाकर्षण की खोज की ओर ले गया!",
                            "tone": "friendly"
                        }
                    ]
                }"""
            elif grade < 9:
                return """
                {
                    "topic": "विज्ञान",
                    "grade_level": "माध्यमिक विद्यालय",
                    "facts": [
                        {
                            "fact": "विज्ञान हमारे चारों ओर की दुनिया को समझने का एक तरीका है। यह हमें बताता है कि चीजें कैसे काम करती हैं, क्यों होती हैं, और कैसे हम अपने आसपास की दुनिया को बेहतर बना सकते हैं। वैज्ञानिक हमेशा नए प्रश्न पूछते हैं और उनके उत्तर खोजने का प्रयास करते हैं।",
                            "tone": "chat"
                        },
                        {
                            "fact": "प्रयोग विज्ञान का एक महत्वपूर्ण हिस्सा हैं। जब हम कोई प्रयोग करते हैं, तो हम वास्तव में एक सवाल पूछ रहे होते हैं और उसका जवाब खोजने की कोशिश कर रहे होते हैं। कभी-कभी हमें गलत जवाब मिलते हैं, लेकिन यह भी सीखने का एक हिस्सा है।",
                            "tone": "friendly"
                        }
                    ]
                }"""
            else:
                return """
                {
                    "topic": "वैज्ञानिक पद्धति",
                    "grade_level": "उच्च विद्यालय",
                    "facts": [
                        {
                            "fact": "वैज्ञानिक पद्धति एक व्यवस्थित प्रक्रिया है जिसका उपयोग वैज्ञानिक अवलोकन, परिकल्पना निर्माण, प्रयोग, और निष्कर्ष निकालने के लिए करते हैं। यह पद्धति वैज्ञानिक खोजों की आधारशिला है और इसने मानव ज्ञान को आगे बढ़ाने में महत्वपूर्ण भूमिका निभाई है।",
                            "tone": "naration-professional"
                        },
                        {
                            "fact": "वैज्ञानिक पद्धति में सबसे महत्वपूर्ण है प्रश्न पूछना और उनके उत्तर खोजने के लिए प्रमाण-आधारित दृष्टिकोण अपनाना। यह हमें मिथकों और अंधविश्वासों से दूर रखती है और वास्तविक ज्ञान की ओर ले जाती है।",
                            "tone": "chat"
                        }
                    ]
                }"""
        else:  # English examples
            if grade < 5:
                return """
                {
                    "topic": "Newton",
                    "grade_level": "Elementary school",
                    "facts": [
                        {
                            "fact": "Did you know that Isaac Newton was just like you—super curious about the world? When he was a kid, he always asked questions and wanted to know how things worked. His curiosity helped him make some of the biggest discoveries in science!",
                            "tone": "cheerful"
                        },
                        {
                            "fact": "Have you ever seen an apple fall from a tree? Well, one day, Newton did too! But instead of just picking it up, he wondered why things always fall down. This led him to discover gravity!",
                            "tone": "friendly"
                        }
                    ]
                }"""
            elif grade < 9:
                return """
                {
                    "topic": "Science",
                    "grade_level": "Middle school",
                    "facts": [
                        {
                            "fact": "Science is a way of understanding the world around us. It tells us how things work, why they happen, and how we can make our world better. Scientists are always asking new questions and trying to find answers.",
                            "tone": "chat"
                        },
                        {
                            "fact": "Experiments are an important part of science. When we do an experiment, we're really asking a question and trying to find its answer. Sometimes we get wrong answers, but that's part of learning too!",
                            "tone": "friendly"
                        }
                    ]
                }"""
            else:
                return """
                {
                    "topic": "Scientific Method",
                    "grade_level": "High school",
                    "facts": [
                        {
                            "fact": "The scientific method is a systematic process used by scientists for observation, hypothesis formation, experimentation, and drawing conclusions. This method forms the foundation of scientific discoveries and has played a crucial role in advancing human knowledge.",
                            "tone": "naration-professional"
                        },
                        {
                            "fact": "At its core, the scientific method is about asking questions and taking an evidence-based approach to finding answers. It keeps us away from myths and superstitions and leads us toward real knowledge.",
                            "tone": "chat"
                        }
                    ]
                }"""

    def generate_response(
        self, 
        topic: str, 
        grade_level: str, 
        language: str = "English",
        model: str = "openai/gpt-4o-mini"
    ) -> Optional[str]:
        """
        Generate structured and engaging facts about the topic for the specified grade level and language.
        
        Args:
            topic (str): The topic to generate facts about
            grade_level (str): The grade level (1-12)
            language (str): The language to generate facts in (default: "English")
            model (str): The model to use for generation
            
        Returns:
            Optional[str]: Generated facts in the specified language or None if an error occurs
        """
        try:
            grade = int(grade_level)
            lang = language.lower()
            
            if lang not in self._language_templates:
                print(f"Warning: Language '{language}' not found in templates. Falling back to English.")
                lang = "english"
            
            # Get language-specific template
            template = self._language_templates[lang]
            
            # Get example based on grade level and language
            example = self._get_example_by_grade(grade, language)
            
            # Construct prompt
            prompt = f"""
            System Message:
            {template['system_msg']}
            
            Important Note:
            Keep the JSON structure fields ("topic", "grade_level", "facts", "fact", "tone") in English, 
            but generate the content in {language}.
            
            User Prompt:
            "Generate structured and engaging facts about {topic} for Grade {grade_level} students in {language}."
            
            Example Output:
            {example}
            
            Now generate three or four facts and their tones for the topic "{topic}" appropriate for Grade {grade_level} in {language}.
            Remember to keep JSON field names in English but content in {language}.
            """

            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": template['system_msg']
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None


def text_to_speech(text: str, tone: str, voice: str = 'en-US-NancyNeural', output_file: str = "output.wav") -> None:
    try:
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        ssml = f'''
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice}">
                <mstts:express-as style="{tone}" styledegree="2">
                    {text}
                </mstts:express-as>
            </voice>
        </speak>
        '''
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"Speech saved to {output_file}")
        else:
            logger.error(f"Error synthesizing audio: {result.reason}")
            raise HTTPException(status_code=500, detail="Failed to synthesize audio")
    
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in text_to_speech: {str(e)}")


def respons_to_tts(response):
    texts=[]
    tones=[]
    result_json = json.loads(response)
    for i in range(len(result_json['facts'])):
        texts.append(result_json['facts'][i]['fact'])
        tones.append(result_json['facts'][i]['tone'])
    return texts,tones

# above function response_to_tts is to extract only text and tone from the responese 


@app.post("/generate-facts", response_model=FactResponse)
async def generate_facts(request: FactRequest):
    try:
        logger.info(f"Generating facts for topic: {request.topic}, grade level: {request.grade_level}")
        
        # Step 1: Generate facts using OpenAI
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key is not set")
        
        language = request.language
        response_gen = ResponseGenerator(api_key=OPENAI_API_KEY)
        generated_facts = response_gen.generate_response(request.topic, str(request.grade_level),language)
        
        if not generated_facts:
            raise HTTPException(status_code=500, detail="Failed to generate facts")
        
        texts, tones = respons_to_tts(generated_facts)
        
        # Step 2: Call the /convert-to-audio endpoint asynchronously
        async with httpx.AsyncClient() as client:
            audio_response = await client.post(
                f"{FastAPI_URL}/convert-to-audio",
                json={"text": texts, "tone": tones,"language":language}  
            ) #here now we are passing the language as well to the convert-to-audio endpoint
        
        if audio_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to convert text to audio")
        
        audio_data = audio_response.json()
        audio_urls = audio_data.get("audio_url")

        # Step 3: Return the facts and audio URLs
        return FactResponse(
            facts=texts,
            grade_level=request.grade_level,
            timestamp=datetime.now().isoformat(),
            audio_urls=audio_urls,
            tones=tones
        )
    
    except Exception as e:
        logger.error(f"Error generating facts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating facts: {str(e)}")  


@app.post("/convert-to-audio", response_model=AudioResponse)
async def convert_to_audio(request: AudioRequest):
    try:
        logger.info(f"Converting text to audio. Text length: {len(request.text)} characters")

        # Available voices
        voices_dict = {
            "Arabic": "ar-EG-ShakirNeural",
            "Tamil": "ta-LK-KumarNeural",
            "English": "en-US-JasonNeural",
            "Hindi": "hi-IN-AaravNeural",
            "Malayalam": "ml-IN-SobhanaNeural",
            "Bengali": "bn-IN-TanishaaNeural"
        }

        lang = request.language
        voice = voices_dict.get(lang)  # so based on the language we get from request we only generate voice for that language

        if not voice:
            raise ValueError(f"Voice for language '{lang}' not found.")

        # Ensure the audio_files directory exists
        os.makedirs("audio_files", exist_ok=True)

        audio_files = []  # Store all generated audio file URLs

        for i, (text, tone) in enumerate(zip(request.text, request.tone)):
            language_region = "-".join(voice.split('-')[:2])  # Extract language-region
            audio_filename = f"{language_region}_{i}.wav"
            audio_file_path = os.path.join("audio_files", audio_filename)

            # Convert text to speech
            text_to_speech(text, tone, voice=voice, output_file=audio_file_path)

            # Store the generated audio URL
            audio_files.append(f"{FastAPI_URL}/audio_files/{audio_filename}")

        
        return AudioResponse(
            audio_url=audio_files,  # Return all audio file URLs
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error converting text to audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error converting text to audio: {str(e)}")

# Mount static directory for audio files
from fastapi.staticfiles import StaticFiles
app.mount("/audio_files", StaticFiles(directory="audio_files"), name="audio_files")

