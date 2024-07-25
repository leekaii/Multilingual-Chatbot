import os
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from PIL import Image
import pyttsx3
import speech_recognition as sr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from indicnlp.tokenize import indic_tokenize
from gtts import gTTS
import pygame
import io
import fitz 

os.environ["OPENAI_API_KEY"] = "xyz"


css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
  height: 100%;
}
.chat-message .avatar img {
  max-width: 100%;
  max-height: 100%;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/F5GHVxW/ddayaa.png" alt="ddayaa" border="0">
    </div>   
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src= "https://i.ibb.co/R0b7q46/icons8-user-48.png" alt="icons8-user-48" border="0">
    </div>   
    <div class="message">{{MSG}}</div>
</div>
'''


def get_pdf_text(pdf, language):
    if pdf is not None:
        doc = fitz.open(pdf)
        text = ""
        for page in doc:
            page_text = page.get_text()
            if language != "english":
                # Tokenize text for non-English languages
                if language == "malayalam":
                    tokenized_text = indic_tokenize.trivial_tokenize(page_text, lang="ml")
                elif language == "hindi":
                    tokenized_text = indic_tokenize.trivial_tokenize(page_text, lang="hi")
                elif language == "kannada":
                    tokenized_text = indic_tokenize.trivial_tokenize(page_text, lang="kn")
                else:
                    tokenized_text = page_text  # If language is not recognized, use original text
                text += " ".join(tokenized_text)
            else:
                text += page_text
    return text

    # if pdf is not None:
    #     pdf_reader = PdfReader(pdf)
    #     text = ""
    #     if language == "english":
    #         for page in pdf_reader.pages:
    #             text += page.extract_text()
    #     else:
    #         for page in pdf_reader.pages:
    #             page_text = page.extract_text()
    #             if language == "malayalam":
    #                 tokenized_text += " ".join(indic_tokenize.trivial_tokenize(page.get_text(), lang="ml"))
    #             elif language == "hindi":
    #                 tokenized_text += " ".join(indic_tokenize.trivial_tokenize(page.get_text(), lang="hi"))
    #             elif language == "kannada":
    #                 tokenized_text += " ".join(indic_tokenize.trivial_tokenize(page.get_text(), lang="kn"))
    #             else:
    #                 tokenized_text = page_text  # If language is not recognized, use original text
    #             text += " ".join(tokenized_text)

    # return text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,  # helps to split equally
            length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def  get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
    memory_key='chat_history',return_messages=True)

    conservation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        memory=memory)
    return conservation_chain


def handle_userinput(user_question, language):
    if language in ["english", "malayalam", "hindi", "kannada"]:
        response = st.session_state.conversation({'question': user_question})

        if st.session_state.chat_history is None:
             st.session_state.chat_history = []

        formatted_response = {
            'question': user_question,
            'response': response
        }

        return formatted_response

    else:
        st.write("Sorry, this language is not supported.")
        return ""


def speech_to_text(language):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

    try:
        if language == "english":
            text = r.recognize_google(audio)
        elif language == "malayalam":
            text = r.recognize_google(audio, language='ml')
        elif language == "hindi":
            text = r.recognize_google(audio, language='hi')
        elif language == "kannada":
            text = r.recognize_google(audio, language='kn')
        else:
            st.write("Sorry, this language is not supported.")
            return ""
        
    except sr.UnknownValueError:
        if language == "english":
            st.write("Can you repeat?")
        elif language == "malayalam":
            st.write("നിങ്ങൾക്ക് വീണ്ടും ആവർത്തിക്കാമോ?")
        elif language == "hindi":
            st.write("क्या आप दोबारा दोहरा सकते हैं?")
        elif language == "kannada":
            st.write("ನೀವು ಮತ್ತೆ ಪುನರಾವರ್ತಿಸಬಹುದೇ?")
        return ""
    except sr.RequestError as e:
        st.write(f"Error fetching results; {e}")
        return ""

    #st.write(text.lower())
    return text.lower()

def text_to_speech(text, language):
    voice_mapping = {
        'english': ('en'),
        'hindi': ('hi'),
        'malayalam': ('ml'),
        'kannada': ('kn')
    }

    selected_voice = voice_mapping.get(language.lower(), 'en')

    if selected_voice == 'en':
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        tts = gTTS(text=text, lang=selected_voice)
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


def english_page():
   image_path = r"C:\projects\Dayabot\ddayaa.png"
   image = Image.open(image_path)
        
   st.write("")
   left_co, cent_co, last_co = st.columns(3)
   with cent_co:
        st.image(image, width=200, output_format="PNG")

   st.markdown(
        """
        <div style="text-align:center">
            <h2>How can I help you?</h2>
            <p>Connection in Every Conversation 🤝</p>
        </div>
        """,
        unsafe_allow_html=True)
   

def malayalam_page():
    image_path = r"C:\projects\Dayabot\ddayaa.png"
    image = Image.open(image_path)
    
    st.write("")
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
            st.image(image, width=200, output_format="PNG")

    st.markdown(
        """
        <div style="text-align:center">
            <h3>എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാനാകും?</h3>
            <p>ഓരോ സംഭാഷണത്തിലും ബന്ധം🤝</p>
        </div>
        """,
        unsafe_allow_html=True)
    

def kannada_page():
     
    image_path = r"C:\projects\Dayabot\ddayaa.png"
    image = Image.open(image_path)
    st.write("")
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
            st.image(image, width=200, output_format="PNG")

    st.markdown(
        """
        <div style="text-align:center">
            <h2>ನಾನು ನಿನಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?</h2>
            <p>ಪ್ರತಿ ಸಂಭಾಷಣೆಯಲ್ಲಿ ಸಂಪರ್ಕ 🤝</p>
        </div>
        """,
        unsafe_allow_html=True)
    

def hindi_page():
     image_path = r"C:\projects\Dayabot\ddayaa.png"
     image = Image.open(image_path)
     st.write("")
     left_co, cent_co, last_co = st.columns(3)
     with cent_co:
            st.image(image, width=200, output_format="PNG")
       
     st.markdown(
        """
        <div style="text-align:center">
            <h2>मैं आपकी कैसे मदद कर सकता हूँ?</h2>
            <p>हर बातचीत में संबंध🤝</p>
        </div>
        """,
        unsafe_allow_html=True)
     


def main():

    load_dotenv()

    st.set_page_config(page_title="Dayabot")
   
    selected = option_menu(
        menu_title = None,
        options =["English", "മലയാളം", "ಕನ್ನಡ", "हिंदी"],
        orientation="horizontal"
    )
   
    if selected == "English":
        english_page()
        mic_button = st.sidebar.button("Ask DayaBot 🎙️")
        user_question = st.chat_input("Start your conversation with DayaBot..")
        pdf = r"C:\projects\Dayabot\pdfs\english.pdf"
        language = "english"


    
    elif selected == "മലയാളം":
        malayalam_page()
        mic_button = st.sidebar.button("ദയബോട്ടിനോട് ചോദിക്കൂ 🎙️")
        user_question = st.chat_input("ദയാബോട്ട് മായി നിങ്ങളുടെ സംഭാഷണം ആരംഭിക്കുക..")
        pdf = r"C:\projects\Dayabot\pdfs\mal_pdf.pdf"
        language = "malayalam"    

       
    elif selected == "ಕನ್ನಡ":
       kannada_page()
       mic_button = st.sidebar.button("ದಯಾಬೋಟ್ ಕೇಳಿ 🎙️")
       user_question = st.chat_input("ದಯಾಬೋಟ್ ನೊಂದಿಗೆ ನಿಮ್ಮ ಸಂವಾದವನ್ನು ಪ್ರಾರಂಭಿಸಿ..")
       pdf = r"C:\projects\Dayabot\pdfs\kan_pdf.pdf"
       language = "kannada"
       
    elif selected == "हिंदी":
       hindi_page()
       mic_button = st.sidebar.button("दयाबोट से पूछो 🎙️")
       user_question = st.chat_input("दयाबॉट के साथ अपनी बातचीत शुरू करें..")
       pdf = r"C:\projects\Dayabot\pdfs\hindi_pdf.pdf"
       language = "hindi"
    else:
        pdf = None
    

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.mic_input = False

    if pdf is not None:
        raw_text = get_pdf_text(pdf,language)
        text_chunks = get_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation(vectorstore)

    if user_question:
        response = handle_userinput(user_question, language)
        user_role = "User"
        bot_role = "DayaBot"
        user_text = user_question
        bot_text = response['response']['answer']  # Assuming 'answer' contains the bot's response
        st.session_state.chat_history.append({"role": user_role, "text": user_text})
        st.session_state.chat_history.append({"role": bot_role, "text": bot_text})
        text_to_speech(bot_text, language)

    if mic_button:
        st.session_state.mic_input = True

    if st.session_state.mic_input:
        user_question = speech_to_text(language)
        if user_question:
            response = handle_userinput(user_question, language)
            #st.write(response)
       
            #if "DayaBot" in response['response']['answer']:
            st.session_state.chat_history.append({"role": "User", "text": user_question})
            st.session_state.chat_history.append({"role": "DayaBot", "text": response['response']['answer']})
            text_to_speech(response['response']['answer'], language)
            st.session_state.mic_input = False
    

    st.markdown(css, unsafe_allow_html=True)
# Display the conversation
    for message in st.session_state.chat_history:
        if message["role"] == "User":
           st.markdown(user_template.replace("{{MSG}}", message["text"]), unsafe_allow_html=True)
        elif message["role"] == "DayaBot":
           st.markdown(bot_template.replace("{{MSG}}", message["text"]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()