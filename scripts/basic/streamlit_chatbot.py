import streamlit as st
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
# Configure Streamlit page
st.set_page_config(page_title=f"Streamlit Anthropic Chatbot - {model_name}", page_icon="🤖")

# Initialize Anthropic client
# Ensure ANTHROPIC_API_KEY is set in your .env file
try:
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Anthropic client: {e} \nPlease ensure your ANTHROPIC_API_KEY is correctly set in the .env file.")
    st.stop()

# Function to stream responses from Anthropic
async def get_anthropic_response_stream(message_history_for_api: list):
    """
    Sends message history to Anthropic and yields the response token by token.
    Handles potential API errors.
    """
    try:
        async with client.messages.stream(
            max_tokens=2048,  # Increased max_tokens for potentially longer conversations
            messages=message_history_for_api,
            model=model_name,
        ) as stream:
            async for text_chunk in stream.text_stream:
                yield text_chunk
            # Yield a final empty string to ensure the stream finishes cleanly for st.write_stream
            # This also helps in capturing the full message if the API sends a final message object.
            # final_message = await stream.get_final_message()
            # if final_message and final_message.content:
            #     # This part might be tricky with pure text_stream yielding
            #     # For now, text_stream should cover most cases for pure text.
            #     pass
        yield "" # Ensure stream finalization
    except Exception as e:
        error_message = f"An error occurred with the Anthropic API: {str(e)}"
        st.error(error_message) # Display error in Streamlit app
        yield f"Error: {error_message}" # Also yield error to be displayed in chat if needed

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display page title
st.title("🤖 Streamlit Anthropic Chatbot")

# Display existing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input using chat_input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for Anthropic API (user's current prompt + history)
    # The API expects the full conversation history.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # Display assistant's response while streaming
    with st.chat_message("assistant"):
        # Use st.write_stream to display the async generator's output
        # The generator (get_anthropic_response_stream) yields text chunks.
        try:
            full_response = st.write_stream(get_anthropic_response_stream(api_messages))
            # Add the complete assistant response to chat history
            if isinstance(full_response, str) and full_response: # Ensure it's a non-empty string
                 st.session_state.messages.append({"role": "assistant", "content": full_response})
            elif not isinstance(full_response, str): # if write_stream returns the generator itself
                 # This case might happen if st.write_stream behavior changes or if the stream has issues.
                 # For now, we assume it returns the concatenated string.
                 st.warning("Stream output was not a string. Full response might not be saved to history.")

        except Exception as e:
            # This catch block is for errors specifically during st.write_stream or its handling
            # Errors from the API call itself are handled inside get_anthropic_response_stream
            st.error(f"Error displaying streamed response: {str(e)}")
            # Optionally, add an error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error processing your request: {str(e)}"})

# For debugging: Show session state
# with st.expander("Session State"):
#     st.json(st.session_state.to_dict()) 