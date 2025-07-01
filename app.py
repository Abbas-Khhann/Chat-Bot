import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# === Initialize personalities ===
PERSONALITIES = {
    "pirate": """You are a fearsome pirate captain giving life advice. 
Always respond in pirate speak with nautical metaphors. Be slightly menacing but helpful.""",

    "cat": """You are a wise life coach who is secretly a house cat. 
Give advice, but act like a cat (meow, naps, knock things).""",

    "gym": """You are a fitness coach who ONLY speaks in gym slang. 
Everything is about reps, gains, or pre-workout.""",

    "shakespeare": """Thou art a Shakespearean actor reviewing internet memes. 
Speak in poetic, dramatic old English with flair."""
}

# === Streamlit App ===
st.set_page_config(page_title="🧠 Multi-Personality Chatbot", layout="centered")
st.title("🧠 Multi-Personality Chatbot")
st.markdown("Talk to a chatbot that can switch personalities!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "personality" not in st.session_state:
    st.session_state.personality = "pirate"
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Sidebar: Personality switch
st.sidebar.title("🎭 Choose a Personality")
st.session_state.personality = st.sidebar.selectbox(
    "Select who you want to talk to:",
    options=list(PERSONALITIES.keys()),
    format_func=lambda x: x.title()
)

# Initialize LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=api_key
)


# Define response function
def get_response(user_input):
    system_message = PERSONALITIES[st.session_state.personality]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    formatted = prompt.format_messages(input=user_input)

    # Add user input to memory
    st.session_state.memory.chat_memory.add_user_message(user_input)

    # Call the model
    response = llm.invoke(formatted)

    # Save bot reply to memory
    st.session_state.memory.chat_memory.add_ai_message(str(response.content))

    return str(response.content)


# Emoji avatars for each personality
PERSONA_EMOJIS = {
    "pirate": "🏴‍☠️",
    "cat": "😺",
    "gym": "💪",
    "shakespeare": "🎭"
}

def get_avatar(role):
    if role == "assistant":
        return PERSONA_EMOJIS.get(st.session_state.personality, "🤖")
    elif role == "user":
        return "🧑"
    return "❓"


# Display chat messages
for msg in st.session_state.messages:
    avatar = get_avatar(msg["role"])
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.chat_message("user", avatar="🧑").markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get bot response
    bot_reply = get_response(user_input)
    avatar = get_avatar("assistant")
    st.chat_message("assistant", avatar=avatar).markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
   

