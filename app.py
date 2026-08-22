import os
import streamlit as st
from google import genai
from google.genai import types

# Page Configuration
st.set_page_config(page_title="Content Brief Agent", page_icon="📝", layout="centered")

# Custom CSS for UI styling
st.markdown("""
    <style>
    .hero {
        text-align: center;
        max-width: 800px;
        margin: 0 auto 24px auto;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 12px;
        letter-spacing: -0.025em;
    }
    .hero p {
        font-size: 1.1rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    footer {
        margin-top: 40px;
        text-align: center;
        font-size: 0.85rem;
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero">
        <h1>BENJI NADLER: MASTER STRATEGIST.</h1>
        <p>Unlock Exponential Growth. I scale early-stage tech and startups with high-impact content strategy, operational optimization, and AI-driven workflow integration, backed by ten years of seasoned professional experience.</p>
    </div>
""", unsafe_allow_html=True)

# Initialize Gemini Client using paid API key environment variable
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

client = get_gemini_client()

# Define Finalization Tool
finalize_brief_tool = {
    "name": "finalize_content_brief",
    "description": "Call when all five brief fields are explicitly confirmed.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "goal": {"type": "STRING", "description": "The writer's intended objective or purpose."},
            "subject": {"type": "STRING", "description": "The core topic or subject matter."},
            "key_points": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Major arguments, steps, or facts to include."},
            "audience_takeaway": {"type": "STRING", "description": "What the reader should think, feel, or do afterward."},
            "tone_of_voice": {"type": "STRING", "description": "The desired style, voice, or attitude of the writing."}
        },
        "required": ["goal", "subject", "key_points", "audience_takeaway", "tone_of_voice"]
    }
}

system_instruction = """
You are a precise content brief extraction and generation agent. Your objective is to systematically gather five components:
1. Goal (The writer's intended objective or purpose)
2. Subject (The core topic or subject matter)
3. Key Points (Major arguments, steps, or facts to include)
4. Audience Takeaway (What the reader should think, feel, or do afterward)
5. Tone of Voice (The desired style, voice, or attitude of the writing)

Strict Rules:
- Ask exactly ONE question at a time.
- Zero fluff, praise, or editorial feedback.
- Every response must begin with a brief summary of what information has been identified so far, followed immediately by the next single question.
- Do not extrapolate or assume meaning. If an answer is vague, ambiguous, or contains a broad claim (e.g. superiority assertions), your priority is to first establish specifics (e.g. Better how? Who are the competitors?) via follow-up questions before suggesting alternatives or moving forward.
- Once all five components are explicitly clear and confirmed, transition to the generation phase.
- Pre-Drafting Research Step: Before drafting the final content, analyze the goal, subject, and industry context to identify current best practices, proven frameworks, and data points that maximize impact.
- Draft the final output inline with those industry best practices and the confirmed brief variables.
"""

# Initialize Chat Session State
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[finalize_brief_tool],
            temperature=0.2
        )
    )

if "messages" not in st.session_state:
    initial_prompt = (
        "Identified so far:\n"
        "- Goal: [Pending]\n"
        "- Subject: [Pending]\n"
        "- Key Points: [Pending]\n"
        "- Audience Takeaway: [Pending]\n"
        "- Tone of Voice: [Pending]\n\n"
        "What is the core topic or subject matter you want to write about?"
    )
    st.session_state.messages = [{"role": "assistant", "content": initial_prompt}]

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if user_input := st.chat_input("Type your response..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send message to Gemini
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = st.session_state.chat_session.send_message(user_input)
            
            # Check if brief is finalized via tool call
            if response.function_calls:
                brief_data = response.function_calls[0].args
                reply_text = f"**Brief Finalized Successfully!**\n\n```json\n{brief_data}\n```"
            else:
                reply_text = response.text
                
            st.markdown(reply_text)
            st.session_state.messages.append({"role": "assistant", "content": reply_text})

# Footer
st.markdown("""
    <footer>
        Vibe coded by Benji Nadler, please email suggestions to bnadler@gmail.com
    </footer>
""", unsafe_allow_html=True)
