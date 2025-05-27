import streamlit as st
import os
from anthropic import AsyncAnthropic, APIStatusError, APIConnectionError, RateLimitError, APIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Constants ---
DEFAULT_MODEL = "claude-3-5-sonnet-20240620" # More standard default model
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

COSTS_PER_MILLION_TOKENS = {
    "cache_creation_input_tokens": 3.75,
    "cache_read_input_tokens": 0.30,
    "input_tokens": 3.00,  # Standard input if not fitting cache categories
    "output_tokens": 15.00,
}

# Configure Streamlit page
st.set_page_config(page_title=f"Chatbot (Tokens & Cost) - {MODEL_NAME}", page_icon="🪙")

# --- Anthropic Client Initialization ---
try:
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Anthropic client: {e}. "
             f"Please ensure ANTHROPIC_API_KEY is correctly set in .env file.")
    st.stop()

# --- Core API Interaction Function ---
async def get_anthropic_response_stream_with_usage(message_history_for_api: list):
    """
    Sends message history to Anthropic, enables caching on the last assistant message if applicable,
    yields response tokens. Usage data and full response text are stored in st.session_state.
    """
    full_response_text = ""
    st.session_state._current_stream_usage_data = None
    st.session_state._current_stream_full_text = ""

    # Prepare messages for API, potentially adding cache_control to the last assistant message
    # This is a simplified approach for multi-turn. For complex scenarios, more elaborate cache point selection might be needed.
    api_messages_with_cache_control = []
    if message_history_for_api:
        api_messages_with_cache_control = [msg.copy() for msg in message_history_for_api] # Deep copy to avoid modifying original session state
        
        # Try to mark the last assistant message's content for caching
        # This is an attempt to cache the conversation up to this point.
        # The effectiveness depends on how the API interprets this for subsequent calls.
        # We only do this if there's more than one message (i.e., at least one user and one assistant turn)
        if len(api_messages_with_cache_control) > 1: 
            last_message_index = len(api_messages_with_cache_control) - 1
            # We are about to get a new assistant message. The history sent will end with the latest user message.
            # To cache the conversation *before* the new user query, we'd ideally mark the *previous* assistant message.
            # However, the cache_control is applied to the prompt *sent*. 
            # Let's try marking the last content block of the *current user's message* if it's not the first message overall.
            # This is more aligned with the idea of caching the prefix *including* the current query if it's built upon a history.
            # A more robust approach might involve a dedicated system prompt or a fixed prefix if the use case allows.

            # For this iteration, let's try adding cache_control to the very last content block of the user's message in the history being sent.
            # This implies we want to cache the history *including* this user turn's message content, if possible.
            # This is not ideal for caching a large static prefix but attempts to use the feature in a conversational flow.
            
            # Let's reconsider: the documentation says "Prompt caching references the entire prompt - tools, system, and messages (in that order) up to and including the block designated with cache_control."
            # To benefit from caching in a conversation, you'd typically mark the last block of an *assistant's* turn, then the next user query would hit that cache.
            # However, when we send the request, the history *ends* with the latest user message.
            # So, for this call, if we want to cache THIS interaction (user query + assistant response), we can't mark *before* the user query easily.
            # The `cache_control` for `client.messages.stream` is on the *input* to that call.

            # Given the error, `cache_control` is NOT a top-level parameter for the stream/create method itself.
            # It needs to be within a content block of the `messages` list.

            # Simplest attempt: mark the last content block of the last message in the history sent to the API.
            # This will be the current user's message.
            if isinstance(api_messages_with_cache_control[-1]["content"], list):
                # If content is a list of blocks, mark the last block
                if api_messages_with_cache_control[-1]["content"]:
                    api_messages_with_cache_control[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            elif isinstance(api_messages_with_cache_control[-1]["content"], str):
                # If content is a string, convert to list of one text block and mark it
                api_messages_with_cache_control[-1]["content"] = [
                    {"type": "text", "text": api_messages_with_cache_control[-1]["content"], "cache_control": {"type": "ephemeral"}}
                ]
            # If there are no messages, or content is not a list/str, this will be skipped.

    else: # No history, this is the first message
        api_messages_with_cache_control = []

    try:
        async with client.messages.stream(
            max_tokens=2048,
            messages=api_messages_with_cache_control if api_messages_with_cache_control else message_history_for_api, # Use modified list if available
            model=MODEL_NAME
            # Removed cache_control from here
        ) as stream:
            async for text_chunk in stream.text_stream:
                full_response_text += text_chunk
                yield text_chunk
            final_message = await stream.get_final_message()
            st.session_state._current_stream_usage_data = final_message.usage
            st.session_state._current_stream_full_text = full_response_text
        yield ""  # Ensure stream finalization for st.write_stream
    except APIConnectionError as e:
        st.error(f"Anthropic API Connection Error: {e.__cause__}")
        st.session_state._current_stream_full_text = full_response_text # Store potentially partial text
        yield f"Error: Connection failed. {e.__cause__}"
    except RateLimitError as e:
        st.error(f"Anthropic API Rate Limit Exceeded: {e}")
        st.session_state._current_stream_full_text = full_response_text
        yield f"Error: Rate limit exceeded. {e}"
    except APIStatusError as e:
        st.error(f"Anthropic API Status Error (code {e.status_code}): {e.response}")
        st.session_state._current_stream_full_text = full_response_text
        yield f"Error: API Status {e.status_code}. {e.response}"
    except APIError as e:
        st.error(f"Generic Anthropic API Error: {e}")
        st.session_state._current_stream_full_text = full_response_text
        yield f"Error: API Error. {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state._current_stream_full_text = full_response_text
        yield f"Error: Unexpected issue. {e}"
    # No return statements with values in an async generator

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_total_usage" not in st.session_state:
    st.session_state.session_total_usage = {}
if "session_total_cost" not in st.session_state:
    st.session_state.session_total_cost = 0.0

# --- UI Rendering --- 
st.title("🤖 Chatbot with Token Counting & Cost")
st.caption(f"Using model: {MODEL_NAME}")

# Display chat messages from history
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "usage" in message and message["usage"]:
            usage = message["usage"]
            cost = message.get("cost", 0.0)
            details = []
            token_types_ordered = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
            for token_type in token_types_ordered:
                if token_type in usage and usage[token_type] > 0:
                    details.append(f"{token_type.replace('_', ' ').title()}: {usage[token_type]}")
            
            if cost > 0:
                details.append(f"Cost: ${cost:.6f}")
            
            if details:
                st.caption(", ".join(details))

# Sidebar for Session Totals
with st.sidebar:
    st.header("Session Statistics")
    st.subheader("Total Token Usage")
    total_usage_display = st.session_state.session_total_usage
    if not total_usage_display:
        st.info("No API calls yet in this session.")
    for token_type, count in total_usage_display.items():
        if count > 0:
            st.metric(label=f"Total {token_type.replace('_', ' ').title()}", value=count)
    
    st.subheader("Total Estimated Cost")
    st.metric(label="Session Cost", value=f"${st.session_state.session_total_cost:.6f}")

# Handle user input
if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        # displayed_response_text will be the concatenation of all strings yielded by the generator
        displayed_response_text = st.write_stream(get_anthropic_response_stream_with_usage(api_messages))
        
        # Retrieve usage_data and full_text that were stored in session_state by the generator
        usage_info = st.session_state.pop('_current_stream_usage_data', None)
        # full_text_from_generator_ss = st.session_state.pop('_current_stream_full_text', None)
        # We will primarily rely on displayed_response_text for the content.

        if displayed_response_text: # This will be non-empty if anything was yielded
            current_cost = 0.0
            # A response is considered successful for cost calculation if usage_info is present.
            # Error messages yielded by the generator won't have usage_info set.
            is_successful_api_response = usage_info is not None

            if is_successful_api_response:
                for token_type, cost_per_mil in COSTS_PER_MILLION_TOKENS.items():
                    tokens_used = getattr(usage_info, token_type, 0)
                    current_cost += (tokens_used / 1_000_000) * cost_per_mil
                
                for token_type in COSTS_PER_MILLION_TOKENS.keys():
                    tokens_in_turn = getattr(usage_info, token_type, 0)
                    st.session_state.session_total_usage[token_type] = \
                        st.session_state.session_total_usage.get(token_type, 0) + tokens_in_turn
                st.session_state.session_total_cost += current_cost

            # Add to message history. 
            # displayed_response_text is what user saw (could be an error message from generator)
            # usage_info and current_cost are based on whether API call was successful before/during get_final_message
            st.session_state.messages.append({
                "role": "assistant",
                "content": displayed_response_text, # This is what st.write_stream rendered
                "usage": usage_info.model_dump() if usage_info else None,
                "cost": current_cost # Will be 0.0 if usage_info was None
            })
            st.rerun()
        else:
            # This case means the generator yielded nothing, or st.write_stream returned None/empty.
            # Check if an error message was already added to history by a previous attempt or if an st.error was shown.
            if not any(msg["role"] == "assistant" and msg["content"].startswith("Error:") for msg in st.session_state.messages[-2:]):
                 st.error("Assistant did not return a message or an error. Please check logs or try again.") 