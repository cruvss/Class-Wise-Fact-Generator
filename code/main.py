
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

# Below we have defined the Pydantic mode for the request and response of the API 
# For fact request we have defined the topic and grade level
# for fact response we have defined the facts, grade level, timestamp and audio_url
# for audio request we have defined the text and tone
# for audio response we have defined the audio_url and timestamp

class FactRequest(BaseModel):
    topic: str
    grade_level: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Newton",
                "grade_level": 5
            }
        }

class FactResponse(BaseModel):
    facts: List[str]
    grade_level: int
    timestamp: str
    audio_url: str  

class AudioRequest(BaseModel):
    text: List[str]
    tone: List[str] 
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": ["Newton discovered gravity."],
                "tone": ["happy"]
            }
        }

class AudioResponse(BaseModel):
    audio_url: str
    timestamp: str

# Class ResponseGenereator is used to generate the response for the given topic and grade level its for fact generation.
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

    def generate_response(self, topic: str, grade_level: str, model: str = "openai/gpt-4o-mini") -> Optional[str]:
        """
        Generate structured and engaging facts about the topic for the specified grade level,
        with tone definitions for TTS (Text-to-Speech) conversion.
        """
        try:
            grade=int(grade_level)
            if grade<5:
                one_shot_example = """
                System Message:
                "You are an expert educator who provides structured facts indeatil in paragrpahs tailored to students of different grade levels. Each fact should be engaging, structured, and appropriate for the given class level. 
                Additionally, define the tone of narration explicitly for each fact so that it can be used for Text-to-Speech (TTS) conversion. 
                Please restrict tones to 'sad', 'depressed', 'hopeful','terrified', or 'serious'."
    
                User Prompt:
                "Generate structured and engaging facts about Newton for a Grade 1 student. Each fact paragraph should include a tone definition that describes how it should be narrated for clarity and engagement."
                
                Example Output 1:
                {
                    "topic": "Newton",
                    "grade_level": "Elementary school",
                    "facts": [
                        {
                            "fact": "Hey there! Did you know that Isaac Newton was just like you—super curious about the world? When he was a kid, he always asked questions and wanted to know how things worked. Instead of just watching things happen, he tried to figure out why they happened! His curiosity helped him make some of the biggest discoveries in science. So if you love asking questions, who knows? Maybe you’ll be the next great scientist too!",
                            "tone": "hopeful"
                        },
                        {
                            "fact": "Have you ever seen an apple fall from a tree? Well, one day, Newton did too! But instead of just picking it up and eating it, he stopped and wondered, ‘Why do things always fall down? Why don’t they float up into the sky?’ He kept thinking about it and discovered something amazing—there’s a force pulling everything toward the ground. He called this force gravity! And guess what? Gravity isn’t just on Earth; it keeps the Moon going around our planet and even makes sure the planets stay in orbit around the Sun!",
                            "tone": "friendly"
                        },
                        {
                            "fact": "Newton didn’t just figure out gravity—he also came up with three really important rules about how things move. Imagine kicking a ball. It keeps rolling until something stops it, right? That’s Newton’s First Law! The Second Law says heavier things need more force to move. That’s why pushing a toy car is easy, but pushing a real car? Nope! And the Third Law? It says that for every action, there’s an equal and opposite reaction. That’s why when you jump, the ground pushes back and you go up! These rules explain everything from riding a bike to how rockets blast into space!",
                            "tone": "calm"
                        },
                        {
                            "fact": "Newton also LOVED rainbows! He wanted to know where colors came from, so he did an experiment with a glass prism. When he shined white light through it—BOOM!—it split into all the colors of the rainbow! That’s when he realized that white light is actually made of many colors mixed together. So the next time you see a rainbow, remember—it’s all thanks to light and Newton’s curiosity!",
                            "tone": "joyful"
                        },
                        {
                            "fact": "Okay, get ready for this—Newton was SO smart that he even invented a whole new kind of math! It’s called calculus, and it helps scientists figure out super tricky things, like how fast something is moving or how planets travel through space. But don’t worry, you don’t need to learn it just yet! Just know that Newton was a total math genius, and his discoveries still help people today!",
                            "tone": "reassuring"
                        },
                        {
                            "fact": "Isaac Newton was born a REALLY long time ago—in 1643! But even though he lived so long ago, his ideas are still used today. His discoveries helped people build airplanes, explore space, and understand how the universe works. Pretty cool, right? It just goes to show that when you ask big questions and try to find the answers, you can change the world—even hundreds of years later!",
                            "tone": "hopeful"
                        }
                    ]        
                }  """

            elif grade<9 and grade>=5:
                one_shot_example = """
                System Message:
                "You are an expert educator who provides structured facts indeatil in paragrpahs tailored to students of different grade levels. Each fact should be engaging, structured, and appropriate for the given class level. 
                Additionally, define the tone of narration explicitly for each fact so that it can be used for Text-to-Speech (TTS) conversion. 
                Please restrict tones to 'sad', 'depressed', 'hopeful','terrified', or 'serious'."
    
                User Prompt:
                "Generate structured and engaging facts about Newton for a Grade 5 student. Each fact paragraph should include a tone definition that describes how it should be narrated for clarity and engagement."
                
                Example Output 1:
                {
                    "topic": "Death",
                    "grade_level": "Middle school",
                    "facts": [
                        {
                            "fact": "Death is a natural part of life, just like birth and growth. Every living thing—plants, animals, and people—has a life cycle, and one day, that cycle comes to an end. Even though it can be hard to think about, it’s something that everyone experiences. Understanding death helps us appreciate life and the time we have with the people we love.",
                            "tone": "thoughtful"
                        },
                        {
                            "fact": "Losing someone we care about is painful, and it’s completely normal to feel sad, confused, or even angry. Grief is a natural response to loss, and everyone experiences it differently. Some people cry, some need time alone, and others find comfort in talking to friends or family. However you feel, remember that it’s okay to express your emotions and take the time you need to heal.",
                            "tone": "empathetic"
                        },
                        {
                            "fact": "Different cultures and religions have different beliefs about what happens after death. Some believe in heaven, reincarnation, or the idea that a person’s energy becomes part of the universe. Others see death as the end of a journey but believe that the impact a person made during their life lives on. No matter what someone believes, one thing is universal—memories and love don’t disappear, and the way someone influenced our lives will always stay with us.",
                            "tone": "serious"
                        },
                        {
                            "fact": "Many cultures around the world have special traditions to remember and honor those who have passed away. In Mexico, Día de los Muertos (Day of the Dead) is a time to celebrate loved ones with food, music, and decorations. In Japan, Obon is a festival where families light lanterns to guide the spirits of their ancestors. These traditions show that even though someone is gone, they are never forgotten.",
                            "tone": "calm"
                        },
                        {
                            "fact": "If you ever feel overwhelmed by loss, it’s important to reach out for support. Talking to a friend, family member, teacher, or counselor can help you process your feelings. Writing, creating art, or engaging in activities you love can also be a way to cope. Grief doesn’t follow a timeline, but over time, the pain becomes easier to carry, and the happy memories start to shine brighter than the sadness.",
                            "tone": "friendly"
                        },
                        {
                            "fact": "Thinking about death reminds us how important it is to make the most of our time. Every moment we spend with friends and family, every kind word, and every good deed leaves a mark. Instead of fearing death, we can focus on living fully, appreciating the present, and spreading kindness. Because in the end, the love and impact we leave behind are what truly matter.",
                            "tone": "hopeful"
                        }
                    ]
                }  """

            elif grade>=9:
                one_shot_example = """
                System Message:
                "You are an expert educator who provides structured facts indeatil in paragrpahs tailored to students of different grade levels. Each fact should be engaging, structured, and appropriate for the given class level. 
                Additionally, define the tone of narration explicitly for each fact so that it can be used for Text-to-Speech (TTS) conversion. 
                Please restrict tones to 'sad', 'depressed', 'hopeful','terrified', or 'serious'."
    
                User Prompt:
                "Generate structured and engaging facts about Newton for a Grade 12 student. Each fact paragraph should include a tone definition that describes how it should be narrated for clarity and engagement."
                
                Example Output 1:
                {
                   
                    {
                        "topic": "Happiness",
                        "grade_level": "High school",
                        "facts": [
                            {
                                "fact": "Happiness isn’t just a single feeling—it’s a mix of emotions, experiences, and perspectives. It can come from achieving goals, spending time with loved ones, or even something as simple as enjoying a quiet moment alone. What makes one person happy might not be the same for someone else, which is why understanding what brings you joy is one of the most valuable things you can learn about yourself.",
                                "tone": "reflective"
                            },
                            {
                                "fact": "Many people think happiness comes from success, money, or having the ‘perfect life,’ but research shows that’s not entirely true. While basic needs and financial stability are important, true happiness often comes from things like meaningful relationships, personal growth, and gratitude. The happiest people aren’t necessarily those who have everything, but those who appreciate what they have and find purpose in their daily lives.",
                                "tone": "thought-provoking"
                            },
                            {
                                "fact": "Science has a lot to say about happiness! Studies show that small, daily habits—like exercising, getting enough sleep, and practicing mindfulness—can significantly boost happiness levels. Even simple things, like smiling or helping someone else, can trigger chemical reactions in the brain that make you feel better. Happiness isn’t just about big life-changing moments; it’s built in the little things we do every day.",
                                "tone": "informative"
                            },
                            {
                                "fact": "Happiness isn’t about feeling great all the time. In fact, constantly chasing happiness can sometimes make people feel worse. Life comes with ups and downs, and experiencing sadness, frustration, or disappointment is completely normal. Instead of trying to avoid negative emotions, true happiness comes from learning how to navigate them and finding meaning even in difficult moments.",
                                "tone": "realistic"
                            },
                            {
                                "fact": "One of the strongest predictors of long-term happiness is connection—having strong, supportive relationships with friends, family, and even a community. Humans are wired for connection, and spending time with people who uplift and support you can make a huge difference in your overall well-being. Happiness grows when it’s shared, so investing in meaningful relationships is one of the best things you can do for yourself.",
                                "tone": "supportive"
                            },
                            {
                                "fact": "At its core, happiness isn’t just about what happens to you—it’s about how you respond to life. The way you think, the perspective you choose, and the meaning you find in your experiences all play a huge role in how happy you feel. It’s not about avoiding struggles but learning to appreciate the journey. Happiness isn’t a destination—it’s a way of living.",
                                "tone": "inspiring"
                            }
                        ]
                }  """             

            # print(one_shot_example)

            prompt = one_shot_example + f"\n\nTopic: {topic}\nGrade Level: {grade_level}\n\nGenerate three or four facts and their tones for the given topic and grade level."


            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert educator who provides structured facts tailored to students of different grade levels. 
                        Each fact should be engaging, structured, and appropriate for the given class level.
                        Additionally, define the tone of narration explicitly for each fact so that it can be used for Text-to-Speech (TTS) conversion. 
                        Please restrict tones to 'sad','thoughtful', 'friendly', 'hopeful','terrified', or 'serious'."""
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

# Text-to-Speech function using Azure TTS
def text_to_speech(texts: List[str], tones: List[str], voice: str = 'en-US-NancyNeural', output_file: str = "output.wav") -> None:
    try:
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file) #output file is the file where the audio will be saved
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        #the role of synthesizer is to convert the text to speech using the given configuration
        
        
        
        ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
        for text, tone in zip(texts,tones):
            ssml += f"""
                <voice name="{voice}">
                    <mstts:express-as style="{tone}" styledegree="2">
                        {text}
                    </mstts:express-as>
                </voice>
            """
        ssml+= "</speak>"
        
        #now the main task of conversion of text to speech with specific tone is done by the ssml which stands for
        #Speech Synthesis Markup Language 
        
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

# This is to extract only text and tone from the responese 


# Endpoint to   generate facts and convert them to audio
@app.post("/generate-facts", response_model=FactResponse)
async def generate_facts(request: FactRequest):
    try:
        logger.info(f"Generating facts for topic: {request.topic}, grade level: {request.grade_level}")
        
        # Step 1: Generate facts using OpenAI
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key is not set")
        
        response_gen = ResponseGenerator(api_key=OPENAI_API_KEY)
        generated_facts = response_gen.generate_response(request.topic, str(request.grade_level))
        
        if not generated_facts:
            raise HTTPException(status_code=500, detail="Failed to generate facts")
        
        texts,tones = respons_to_tts(generated_facts)
        
        # Step 2: Call the /convert-to-audio endpoint asynchronously
        async with httpx.AsyncClient() as client:
            audio_response = await client.post(
                f"{FastAPI_URL}/convert-to-audio",
                json={"text": texts, "tone": tones}  
            )
        
        if audio_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to convert text to audio")
        
        audio_data = audio_response.json()
        audio_url = audio_data.get("audio_url")

        # Step 3: Return the facts and audio URL
        return FactResponse(
            facts= texts,
            grade_level=request.grade_level,
            timestamp=datetime.now().isoformat(),
            audio_url=audio_url
        )
    
    except Exception as e:
        logger.error(f"Error generating facts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating facts: {str(e)}")
    

    
# Endpoint to convert text to audio
@app.post("/convert-to-audio", response_model=AudioResponse)
async def convert_to_audio(request: AudioRequest):
    try:
        logger.info(f"Converting text to audio. Text length: {len(request.text)} characters")
        
        voices = [ 'en-US-AriaNeural','en-IN-NeerjaNeural', 'ar-AE-FatimaNeural', 'ta-IN-PallaviNeural']
        # we will convert the text to audio into 5 different voices/accents 

        # Ensure the audio_files directory exists
        os.makedirs("audio_files", exist_ok=True)

        audio_files = [] #all the audio file Url will be stored here

        for i, voice in enumerate(voices, start=1):
            language_region = "-".join(voice.split('-')[:2])


            audio_filename = f"{language_region}_{i}.wav"
            audio_file_path = os.path.join("audio_files", audio_filename)

            # Convert text to speech
            text_to_speech(request.text, request.tone, voice=voice, output_file=audio_file_path)

            # Store the generated audio URL
            audio_files.append(f"{FastAPI_URL}/audio_files/{audio_filename}")
        
        return AudioResponse(
            audio_url=audio_files[0],   # but we will return only the first audio file URL others will be stored in audio_files
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error converting text to audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error converting text to audio: {str(e)}")

# Mount static directory for audio files
from fastapi.staticfiles import StaticFiles
app.mount("/audio_files", StaticFiles(directory="audio_files"), name="audio_files")

