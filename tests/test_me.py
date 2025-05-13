# import pytest
# import json
# from unittest.mock import patch, MagicMock, mock_open
# import numpy as np
# import faiss
# from src.main.python.classes import Me, SimpleRAG, Document

# @pytest.fixture
# def mock_openai():
#     """Fixture to mock OpenAI client."""
#     with patch('openai.OpenAI') as mock_openai:
#         mock_instance = MagicMock()
#         mock_openai.return_value = mock_instance
#         yield mock_instance

# @pytest.fixture
# def mock_pdfreader():
#     """Fixture to mock PdfReader."""
#     with patch('src.main.python.classes.PdfReader') as mock_pdfreader:
#         mock_instance = MagicMock()
#         mock_page = MagicMock()
#         mock_page.extract_text.return_value = "Mock LinkedIn profile text"
#         mock_instance.pages = [mock_page]
#         mock_pdfreader.return_value = mock_instance
#         yield mock_pdfreader

# @pytest.fixture
# def mock_file_operations():
#     """Fixture to mock file operations."""
#     summary_content = "Mock summary text"
#     m = mock_open(read_data=summary_content)
#     with patch('builtins.open', m):
#         yield m

# @pytest.fixture
# def mock_rag():
#     """Fixture to mock SimpleRAG."""
#     with patch('src.main.python.classes.SimpleRAG') as mock_rag:
#         mock_instance = MagicMock()
#         mock_instance.search.return_value = [
#             {
#                 "text": "Mock search result 1",
#                 "metadata": {"source": "summary"},
#                 "score": 0.95
#             }
#         ]
#         mock_instance.format_context.return_value = "From summary:\nMock search result 1\n"
#         mock_rag.return_value = mock_instance
#         yield mock_instance

# @pytest.fixture
# def mock_tools():
#     """Fixture to mock tool functions."""
#     with patch('src.main.python.classes.record_user_details') as mock_record_user, \
#          patch('src.main.python.classes.record_unknown_question') as mock_record_question, \
#          patch('src.main.python.classes.send_push_when_engaged') as mock_send_push:
        
#         mock_record_user.return_value = {"recorded": "ok"}
#         mock_record_question.return_value = {"recorded": "ok"}
#         mock_send_push.return_value = {"sent": "ok"}
        
#         yield {
#             "record_user_details": mock_record_user,
#             "record_unknown_question": mock_record_question,
#             "send_push_when_engaged": mock_send_push
#         }

# class TestMe:
#     def test_init(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag):
#         """Test Me initialization."""
#         # Act
#         me = Me()
        
#         # Assert
#         assert me.name == "Anton Kostov"
#         assert mock_rag.add_documents.called
#         assert mock_rag.save.called
        
#     def test_system_prompt(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag):
#         """Test system_prompt method generates correct prompt."""
#         # Arrange
#         me = Me()
        
#         # Act
#         prompt = me.system_prompt()
        
#         # Assert
#         assert me.name in prompt
#         assert "career, background, skills and experience" in prompt
#         assert "record_user_details tool" in prompt
    
#     def test_handle_tool_call_record_user_details(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag, mock_tools):
#         """Test handle_tool_call method with record_user_details tool."""
#         # Arrange
#         me = Me()
#         tool_call = MagicMock()
#         tool_call.function.name = "record_user_details"
#         tool_call.function.arguments = json.dumps({"email": "test@example.com"})
#         tool_call.id = "call123"
        
#         # Act
#         results = me.handle_tool_call([tool_call])
        
#         # Assert
#         assert len(results) == 1
#         assert results[0]["role"] == "tool"
#         assert results[0]["tool_call_id"] == "call123"
#         assert json.loads(results[0]["content"]) == {"recorded": "ok"}
#         mock_tools["record_user_details"].assert_called_once_with(email="test@example.com")
    
#     def test_handle_tool_call_record_unknown_question(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag, mock_tools):
#         """Test handle_tool_call method with record_unknown_question tool."""
#         # Arrange
#         me = Me()
#         tool_call = MagicMock()
#         tool_call.function.name = "record_unknown_question"
#         tool_call.function.arguments = json.dumps({"question": "Unknown question"})
#         tool_call.id = "call456"
        
#         # Act
#         results = me.handle_tool_call([tool_call])
        
#         # Assert
#         assert len(results) == 1
#         mock_tools["record_unknown_question"].assert_called_once_with(question="Unknown question")
    
#     def test_handle_tool_call_send_push(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag, mock_tools):
#         """Test handle_tool_call method with send_push_when_engaged tool."""
#         # Arrange
#         me = Me()
#         tool_call = MagicMock()
#         tool_call.function.name = "send_push_when_engaged"
#         tool_call.function.arguments = json.dumps({})
#         tool_call.id = "call789"
        
#         # Act
#         results = me.handle_tool_call([tool_call])
        
#         # Assert
#         assert len(results) == 1
#         mock_tools["send_push_when_engaged"].assert_called_once()
    
#     def test_chat_simple_response(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag):
#         """Test chat method when a simple response is returned."""
#         # Arrange
#         me = Me()
#         message = "Hello, tell me about yourself"
#         history = []
        
#         # Mock OpenAI response for simple text response
#         mock_response = MagicMock()
#         mock_choice = MagicMock()
#         mock_message = MagicMock()
#         mock_message.content = "I am Anton Kostov, a software developer."
#         mock_choice.message = mock_message
#         mock_choice.finish_reason = "stop"
#         mock_response.choices = [mock_choice]
#         me.openai.chat.completions.create.return_value = mock_response
        
#         # Act
#         result = me.chat(message, history)
        
#         # Assert
#         assert result == "I am Anton Kostov, a software developer."
#         assert mock_rag.search.called
#         assert mock_rag.format_context.called
        
#     def test_chat_with_tool_calls(self, mock_openai, mock_pdfreader, mock_file_operations, mock_rag, mock_tools):
#         """Test chat method when tool calls are triggered."""
#         # Arrange
#         me = Me()
#         message = "Hello, I'd like to get in touch"
#         history = []
        
#         # Mock OpenAI responses - first with tool call, then with text
#         tool_call_response = MagicMock()
#         tool_call_choice = MagicMock()
#         tool_call_message = MagicMock()
#         tool_call = MagicMock()
#         tool_call.function.name = "send_push_when_engaged"
#         tool_call.function.arguments = json.dumps({})
#         tool_call.id = "call123"
#         tool_call_message.tool_calls = [tool_call]
#         tool_call_choice.message = tool_call_message
#         tool_call_choice.finish_reason = "tool_calls"
#         tool_call_response.choices = [tool_call_choice]
        
#         final_response = MagicMock()
#         final_choice = MagicMock()
#         final_message = MagicMock()
#         final_message.content = "I'd be happy to connect with you."
#         final_choice.message = final_message
#         final_choice.finish_reason = "stop"
#         final_response.choices = [final_choice]
        
#         # Set up the side effect to return different values on each call
#         me.openai.chat.completions.create.side_effect = [tool_call_response, final_response]
        
#         # Act
#         result = me.chat(message, history)
        
#         # Assert
#         assert result == "I'd be happy to connect with you."
#         assert me.openai.chat.completions.create.call_count == 2
#         mock_tools["send_push_when_engaged"].assert_called_once()

