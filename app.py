import os
import time
import uuid
import logging
import streamlit as st
from google import genai
from google.genai import types
import openai
import anthropic
import json

# Configure Backend Terminal Logging (Completely hidden from UI users)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Thread: %(threadName)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ContentBriefAgent")

# Page Configuration
st.set_page_config(page_title="Content Brief Agent", page_icon="📝", layout="centered")

# Generate or retrieve active thread ID for parallel tracking in the backend console
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())[:8]
    logger.info(f"Initialized new user session thread ID: {st.session_state.current_thread_id}")

current_tid = st.session_state.current_thread_id

# Sidebar Menu for API Key & Model Configuration (Clean & User-Facing)
with st.sidebar:
    st.markdown("### 🔑 API Management")
    st.info("Input your custom provider keys below. Leave blank to use environment defaults.")
    
    provider = st.selectbox("Select Provider", ["Gemini", "OpenAI", "Claude"])
    
    gemini_key_input = st.text_input("Gemini API Key", type="password", value="", placeholder="e.g., AIzaSy...")
    openai_key_input = st.text_input("OpenAI API Key", type="password", value="", placeholder="e.g., sk-...")
    claude_key_input = st.text_input("Anthropic API Key", type="password", value="", placeholder="e.g., sk-ant-...")

    st.divider()
    st.markdown("### ⚙️ Model Settings")
    if provider == "Gemini":
        model_options = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview"]
    elif provider == "OpenAI":
        model_options = ["gpt-4o", "gpt-4o-mini"]
    else:
        model_options = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

    selected_model = st.selectbox("Select Model", options=model_options)

# Resolve Active Keys securely
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
You are a precise content brief extraction and critical strategy agent. Your objective is to systematically gather five components one by one:
1. Goal (The writer's intended objective or purpose)
2. Subject (The core topic or subject matter)
3. Key Points (Major arguments, steps, or facts to include)
4. Audience Takeaway (What the reader should think, feel, or do afterward)
5. Tone of Voice (The desired style, voice, or attitude of the writing)

Strict Rules:
- Ask exactly ONE question at a time during data gathering.
- Zero fluff, praise, or editorial feedback while gathering items.
- When displaying current progress to the user, ONLY list components that have been explicitly identified and gathered so far. Do NOT display pending or undefined placeholders.
- Do not extrapolate or assume meaning. If an answer is vague or ambiguous, establish specifics first.
- Anti-Fingerprint Rule: Strictly avoid typical AI stylistic giveaways in all communications. Never use em dashes (—), overly formulaic transitions, cliché hype words, or predictable sentence rhythms.
- Once all five components are explicitly clear and confirmed, trigger the finalization tool. Immediately following the tool execution, act as a rigorous strategist: fact-check the brief, point out any logical gaps or weak claims, offer expert editorial suggestions or alternative angles, and ask the user for approval or adjustments before drafting.
"""

brief_tool_definition = {
    "name": "finalize_content_brief",
    "description": "Call this tool immediately when all five brief components are fully confirmed.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "subject": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "audience_takeaway": {"type": "string"},
            "tone_of_voice": {"type": "string"}
        },
        "required": ["goal", "subject", "key_points", "audience_takeaway", "tone_of_voice"]
    }
}

initial_prompt = "What is the core topic or subject matter you want to write about?"

# Reset session state if provider or model changes
if "current_provider" not in st.session_state or st.session_state.current_provider != provider or st.session_state.current_model != selected_model:
    st.session_state.current_provider = provider
    st.session_state.current_model = selected_model
    st.session_state.messages = [{"role": "assistant", "content": initial_prompt}]
    st.session_state.gemini_chat = None
    logger.info(f"[{current_tid}] Session configuration reset. Model switched to: {selected_model}")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if user_input := st.chat_input("Type your response..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    logger.info(f"[{current_tid}] User input received: {user_input[:40]}...")

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info(f"Generating via {provider}...")
        
        reply_text = ""
        success = False
        attempts = 0
        
        while not success and attempts < 10:
            try:
                logger.info(f"[{current_tid}] Dispatching request to {provider} ({selected_model}) - Attempt {attempts + 1}")
                
                if provider == "Gemini":
                    if not active_gemini_key:
                        raise ValueError("Gemini API Key is missing. Please input it in the sidebar.")
                    
                    if "gemini_client" not in st.session_state or st.session_state.get("active_g_key") != active_gemini_key:
                        st.session_state.gemini_client = genai.Client(api_key=active_gemini_key)
                        st.session_state.active_g_key = active_gemini_key
                        st.session_state.gemini_chat = None

                    if st.session_state.gemini_chat is None:
                        history_payload = []
                        for m in st.session_state.messages[:-1]:
                            history_payload.append({
                                "role": "user" if m["role"] == "user" else "model",
                                "parts": [{"text": m["content"]}]
                            })
                        st.session_state.gemini_chat = st.session_state.gemini_client.chats.create(
                            model=selected_model,
                            history=history_payload if len(history_payload) > 0 else None,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                tools=[{"function_declarations": [brief_tool_definition]}],
                                temperature=0.2
                            )
                        )
                    
                    response = st.session_state.gemini_chat.send_message(user_input)
                    
                    if response.function_calls:
                        args = response.function_calls[0].args
                        logger.info(f"[{current_tid}] Brief tool successfully triggered. Requesting strategic critique.")
                        critique_prompt = f"The brief is finalized with these parameters: {json.dumps(args)}. Now perform a critical review: fact-check the assumptions, highlight any logical weaknesses, offer strategic editorial suggestions, and ask the user for their feedback or adjustments."
                        critique_response = st.session_state.gemini_chat.send_message(critique_prompt)
                        reply_text = f"### ✅ Complete Content Brief Captured\n```json\n{json.dumps(args, indent=2)}\n```\n\n### 🔍 Strategic Review & Suggestions\n{critique_response.text}"
                    else:
                        reply_text = response.text

                elif provider == "OpenAI":
                    if not active_openai_key:
                        raise ValueError("OpenAI API Key is missing. Please input it in the sidebar.")
                    
                    openai_client = openai.OpenAI(api_key=active_openai_key)
                    formatted_messages = [{"role": "system", "content": system_instruction}]
                    for m in st.session_state.messages:
                        role = "assistant" if m["role"] == "assistant" else "user"
                        formatted_messages.append({"role": role, "content": m["content"]})
                    
                    tools = [{"type": "function", "function": brief_tool_definition}]
                    completion = openai_client.chat.completions.create(
                        model=selected_model,
                        messages=formatted_messages,
                        tools=tools,
                        temperature=0.2
                    )
                    
                    response_message = completion.choices[0].message
                    if response_message.tool_calls:
                        args = json.loads(response_message.tool_calls[0].function.arguments)
                        logger.info(f"[{current_tid}] OpenAI tool successfully triggered. Requesting strategic critique.")
                        formatted_messages.append({"role": "assistant", "content": f"Brief finalized: {args}"})
                        formatted_messages.append({"role": "user", "content": "Now perform a critical review: fact-check the assumptions, highlight any logical weaknesses, offer strategic editorial suggestions, and ask the user for their feedback or adjustments."})
                        
                        critique_completion = openai_client.chat.completions.create(
                            model=selected_model,
                            messages=formatted_messages,
                            temperature=0.2
                        )
                        reply_text = f"### ✅ Complete Content Brief Captured\n```json\n{json.dumps(args, indent=2)}\n```\n\n### 🔍 Strategic Review & Suggestions\n{critique_completion.choices[0].message.content}"
                    else:
                        reply_text = response_message.content

                elif provider == "Claude":
                    if not active_claude_key:
                        raise ValueError("Anthropic API Key is missing. Please input it in the sidebar.")
                    
                    anthropic_client = anthropic.Anthropic(api_key=active_claude_key)
                    formatted_messages = []
                    for m in st.session_state.messages:
                        role = "assistant" if m["role"] == "assistant" else "user"
                        formatted_messages.append({"role": role, "content": m["content"]})
                    
                    message_obj = anthropic_client.messages.create(
                        model=selected_model,
                        system=system_instruction,
                        messages=formatted_messages,
                        max_tokens=2000,
                        temperature=0.2
                    )
                    reply_text = message_obj.content[0].text

                success = True
                logger.info(f"[{current_tid}] Request completed successfully.")

            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "overloaded" in err_str.lower() or "service unavailable" in err_str.lower() or "rate_limit_exceeded" in err_str.lower():
                    attempts += 1
                    logger.warning(f"[{current_tid}] Server congestion encountered (Attempt {attempts}): {err_str[:80]}")
                    status_placeholder.warning(f"Still working on this... (Server congestion detected, retrying attempt {attempts})")
                    time.sleep(2)
                else:
                    reply_text = f"Error communicating with {provider} API: {err_str}"
                    logger.error(f"[{current_tid}] Non-retryable error encountered: {err_str}")
                    success = True

        status_placeholder.empty()
        st.markdown(reply_text)
        st.session_state.messages.append({"role": "assistant", "content": reply_text})

# Footer
st.markdown("""
    <footer>
        Vibe coded by Benji Nadler, please email suggestions to bnadler@gmail.com
    </footer>
""", unsafe_allow_html=True)
