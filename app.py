import os
import streamlit as st
from google import genai
from google.genai import types
import openai
import anthropic

# Page Configuration
st.set_page_config(page_title="Content Brief Agent", page_icon="📝", layout="centered")

# Sidebar Menu for API Key & Model Configuration
with st.sidebar:
    st.header("🔑 API Credentials")
    provider = st.selectbox("Select Provider", ["Gemini", "OpenAI", "Claude"])
    
    gemini_key_input = st.text_input("Gemini API Key", type="password", value="", help="Enter your paid Gemini API key.")
    openai_key_input = st.text_input("OpenAI API Key", type="password", value="", help="Enter your OpenAI API key.")
    claude_key_input = st.text_input("Anthropic API Key", type="password", value="", help="Enter your Claude API key.")

    st.divider()
    st.header("⚙️ Model Settings")
    if provider == "Gemini":
        model_options = ["gemini-2.5-flash", "gemini-2.5-pro"]
    elif provider == "OpenAI":
        model_options = ["gpt-4o", "gpt-4o-mini"]
    else:
        model_options = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

    selected_model = st.selectbox("Select Model", options=model_options)

# Resolve Active Keys
active_gemini_key = gemini_key_input if gemini_key_input else os.environ.get("GEMINI_API_KEY", "")
active_openai_key = openai_key_input if openai_key_input else os.environ.get("OPENAI_API_KEY", "")
active_claude_key = claude_key_input if claude_key_input else os.environ.get("ANTHROPIC_API_KEY", "")

# Custom CSS
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
- Every response must begin with a brief summary of what information has been identified so far (tracking items as either [Pending] or showing the gathered value), followed immediately by the next single question.
- Do not extrapolate or assume meaning. If an answer is vague, ambiguous, or contains a broad claim, your priority is to first establish specifics via follow-up questions before moving forward.
- Once all five components are explicitly clear and confirmed, output the final structured brief.
"""

initial_prompt = (
    "Identified so far:\n"
    "- Goal: [Pending]\n"
    "- Subject: [Pending]\n"
    "- Key Points: [Pending]\n"
    "- Audience Takeaway: [Pending]\n"
    "- Tone of Voice: [Pending]\n\n"
    "What is the core topic or subject matter you want to write about?"
)

# Initialize or reset session state if provider/model shifts
if "current_provider" not in st.session_state or st.session_state.current_provider != provider or st.session_state.current_model != selected_model:
    st.session_state.current_provider = provider
    st.session_state.current_model = selected_model
    st.session_state.messages = [{"role": "assistant", "content": initial_prompt}]
    st.session_state.gemini_chat = None

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if user_input := st.chat_input("Type your response..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"Generating via {provider}..."):
            reply_text = ""
            try:
                if provider == "Gemini":
                    if not active_gemini_key:
                        raise ValueError("Gemini API Key is missing. Please enter it in the sidebar menu.")
                    
                    client = genai.Client(api_key=active_gemini_key)
                    
                    # Rebuild or resume the Gemini native chat object keeping full history intact
                    if st.session_state.get("gemini_chat") is None:
                        history_payload = []
                        # Map existing messages into Gemini content types if restarting mid-session
                        for m in st.session_state.messages[:-1]:
                            history_payload.append({
                                "role": "user" if m["role"] == "user" else "model",
                                "parts": [{"text": m["content"]}]
                            })
                        st.session_state.gemini_chat = client.chats.create(
                            model=selected_model,
                            history=history_payload if len(history_payload) > 0 else None,
                            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
                        )
                    
                    response = st.session_state.gemini_chat.send_message(user_input)
                    reply_text = response.text

                elif provider == "OpenAI":
                    if not active_openai_key:
                        raise ValueError("OpenAI API Key is missing. Please enter it in the sidebar menu.")
                    
                    openai_client = openai.OpenAI(api_key=active_openai_key)
                    formatted_messages = [{"role": "system", "content": system_instruction}]
                    for m in st.session_state.messages:
                        # Map assistant -> assistant, user -> user
                        role = "assistant" if m["role"] == "assistant" else "user"
                        formatted_messages.append({"role": role, "content": m["content"]})
                    
                    completion = openai_client.chat.completions.create(
                        model=selected_model,
                        messages=formatted_messages,
                        temperature=0.2
                    )
                    reply_text = completion.choices[0].message.content

                elif provider == "Claude":
                    if not active_claude_key:
                        raise ValueError("Anthropic API Key is missing. Please enter it in the sidebar menu.")
                    
                    anthropic_client = anthropic.Anthropic(api_key=active_claude_key)
                    formatted_messages = []
                    for m in st.session_state.messages:
                        role = "assistant" if m["role"] == "assistant" else "user"
                        formatted_messages.append({"role": role, "content": m["content"]})
                    
                    message_obj = anthropic_client.messages.create(
                        model=selected_model,
                        system=system_instruction,
                        messages=formatted_messages,
                        max_tokens=1000,
                        temperature=0.2
                    )
                    reply_text = message_obj.content[0].text

            except Exception as e:
                reply_text = f"Error communicating with {provider} API: {str(e)}"

            st.markdown(reply_text)
            st.session_state.messages.append({"role": "assistant", "content": reply_text})

# Footer
st.markdown("""
    <footer>
        Vibe coded by Benji Nadler, please email suggestions to bnadler@gmail.com
    </footer>
""", unsafe_allow_html=True)
