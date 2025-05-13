import gradio as gr
from classes import Me
import logging
from utils_and_tools import record_user_details, record_unknown_question, send_push_when_engaged

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting application...")
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
