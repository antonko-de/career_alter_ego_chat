from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from rag import SimpleRAG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Anton Kostov"
        
        # Initialize RAG system with custom chunk size
        logger.info("Initializing RAG system...")
        self.rag = SimpleRAG(chunk_size=300, chunk_overlap=50)
        
        # Load and process LinkedIn profile
        reader = PdfReader("resources/Profile.pdf")
        linkedin_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                linkedin_text += text
        
        # Load summary
        with open("resources/summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()
        
        # Add documents to RAG system with metadata
        logger.info("Adding documents to RAG system...")
        self.rag.add_documents(
            documents=[summary, linkedin_text],
            metadata_list=[
                {"source": "summary", "type": "background"},
                {"source": "linkedin", "type": "professional_profile"}
            ]
        )
        
        # Save the processed documents
        logger.info("Saving RAG data...")
        self.rag.save("rag_data")
        logger.info("Initialization complete!")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        logger.info(f"Received message: {message[:100]}...")
        # Get relevant context from RAG
        results = self.rag.search(message, k=3)
        context = self.rag.format_context(results)
        logger.info(f"Retrieved {len(results)} relevant documents")
        
        # Add context to the message with clear separation
        enhanced_message = f"""Here is the relevant context from my background:

{context}

Based on this context, please answer the following question:
{message}"""
        
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": enhanced_message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        logger.info("Generated response successfully")
        return response.choices[0].message.content
    

if __name__ == "__main__":
    logger.info("Starting application...")
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    