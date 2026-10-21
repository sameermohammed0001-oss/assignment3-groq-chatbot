
import os
import gradio as gr
from groq import Groq

MODEL = "openai/gpt-oss-120b"


def chatbot(message, history):
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return "Error: GROQ_API_KEY is missing from the environment settings."

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI chatbot. "
                "Answer clearly and explain concepts simply."
            )
        }
    ]

    # Add the previous conversation
    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({
                "role": role,
                "content": content
            })

    # Add the current question
    messages.append({
        "role": "user",
        "content": message
    })

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"Error communicating with Groq: {error}"


app = gr.ChatInterface(
    fn=chatbot,
    type="messages",
    title="My AI Chatbot",
    description="Ask questions and get AI-powered answers using Groq."
)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
