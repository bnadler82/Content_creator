from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize client using your paid API key from environment variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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

# Store a session chat map (for production apps, use session middleware or user tokens)
active_chats = {}

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    # Initialize a new chat session when loading the page
    chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[finalize_brief_tool],
            temperature=0.2
        )
    )
    # Give it an initial hidden trigger or let the user start typing
    active_chats["default"] = chat_session
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message")
    
    chat = active_chats.get("default")
    if not chat:
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[finalize_brief_tool],
                temperature=0.2
            )
        )
        active_chats["default"] = chat

    response = chat.send_message(user_message)
    
    # Check if the model triggered the finalization tool
    tool_data = None
    if response.function_calls:
        tool_data = response.function_calls[0].args

    return {
        "response": response.text,
        "completed": bool(tool_data),
        "brief": tool_data
    }
