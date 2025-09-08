import streamlit as st
from youtube_chatbot import YouTubeChatbot
import re

st.title("YouTube Chatbot")

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = YouTubeChatbot()
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

def extract_video_id(url_or_id):
    if 'youtube.com' in url_or_id or 'youtu.be' in url_or_id:
        match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url_or_id)
        return match.group(1) if match else None
    return url_or_id

# Sidebar for video setup
with st.sidebar:
    st.header("Setup")
    video_input = st.text_input("YouTube URL or Video ID:", placeholder="Gfr50f6ZBvo")
    
    if st.button("Initialize", use_container_width=True) and video_input:
        video_id = extract_video_id(video_input)
        if video_id:
            with st.spinner("Processing..."):
                result = st.session_state.chatbot.initialize_chatbot(video_id)
                if "successfully" in result:
                    st.session_state.initialized = True
                    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm ready to answer questions about this video."}]
                    st.success("Ready!")
                else:
                    st.error(result)
        else:
            st.error("Invalid input")

# Chat interface
if st.session_state.initialized:
    # Display chat messages
    for message in st.session_state.messages:
        icon = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=icon):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask_question(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please initialize the chatbot with a YouTube video first.")