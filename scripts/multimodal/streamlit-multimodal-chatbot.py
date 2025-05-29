import os
import base64 # For encoding images
import mimetypes # For determining media type

import streamlit as st
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Constants ---
DEFAULT_MODEL = "claude-3-5-haiku-20241022" # Updated default to Sonnet for better multimodal
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
MAX_TOKENS_OUTPUT = 4096 # Increased max tokens for potentially larger multimodal responses

# Model-specific costs per million tokens
COSTS_PER_MILLION_TOKENS = {
    "claude-3-5-sonnet-20240620": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75, # Cost for creating a cache from input
        "cache_read_input_tokens": 0.30,   # Cost for reading from cache (considered as input)
    },
    "claude-3-opus-20240229": {
        "input_tokens": 15.00,
        "output_tokens": 75.00,
        "cache_creation_input_tokens": 18.75,
        "cache_read_input_tokens": 1.50,
    },
    "claude-3-haiku-20240307": {
        "input_tokens": 0.25,
        "output_tokens": 1.25,
        "cache_creation_input_tokens": 0.30,
        "cache_read_input_tokens": 0.03,
    },
    "claude-3-5-haiku-20241022": { 
        "input_tokens": 0.8, # Assuming costs, adjust if known
        "output_tokens": 4,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 0.08,
    },
    # Older models kept for reference if needed
    "claude-sonnet-4-20250514": { 
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
    "claude-opus-4-20240229": { 
        "input_tokens": 15.00,
        "output_tokens": 75.00,
        "cache_creation_input_tokens": 18.75,
        "cache_read_input_tokens": 1.50,
    },
}

DEFAULT_MODEL_COSTS = COSTS_PER_MILLION_TOKENS.get(
    MODEL_NAME, # Use current model for default costs
    { # Fallback default costs if model not in dict (e.g. a new one)
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
)

st.set_page_config(
    page_title=f"Multimodal Chatbot - {MODEL_NAME}", page_icon="🖼️"
)

# --- Anthropic Client Initialization ---
try:
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    st.error(
        f"Failed to initialize Anthropic client: {e}. "
        f"Please ensure ANTHROPIC_API_KEY is correctly set in .env file."
    )
    st.stop()

# --- Helper Functions ---
def get_model_costs(model_name_to_check):
    """Safely retrieves costs for a given model, falling back to defaults."""
    return COSTS_PER_MILLION_TOKENS.get(model_name_to_check, DEFAULT_MODEL_COSTS)

# --- Core API Interaction Function ---
async def get_anthropic_response_stream_with_usage(
    api_messages_for_call: list, # Renamed for clarity
):
    """
    Sends message history (now potentially multimodal) to Anthropic.
    Enables caching on the last text part of the last assistant message if applicable.
    Yields response tokens. Usage data and full response text are stored in session state.
    Allows for stopping the stream.
    """
    full_response_text = ""
    # These are initialized in session state now, direct assignment for this turn
    st.session_state['_current_stream_usage_data'] = None
    st.session_state['_current_stream_full_text'] = ""
    st.session_state['streaming_in_progress'] = True
    st.session_state['_stream_stopped_by_user_flag'] = False

    # The section that previously modified api_messages_for_call[-2] for caching 
    # assistant's response has been removed as per user request to simplify caching.
    # Caching for user-uploaded media is handled during the construction of api_messages_for_call.

    try:
        async with client.messages.stream(
            max_tokens=MAX_TOKENS_OUTPUT,
            messages=api_messages_for_call, # Directly use the prepared messages
            model=MODEL_NAME,
        ) as stream:
            try:
                async for text_chunk in stream.text_stream:
                    if st.session_state.get("stop_streaming", False): # Check global stop signal
                        st.session_state['_stream_stopped_by_user_flag'] = True
                        break
                    full_response_text += text_chunk
                    yield text_chunk
                
                # Only try to get final message if streaming completed successfully
                if not st.session_state.get("stop_streaming", False):
                    try:
                        final_message = await stream.get_final_message()
                        st.session_state['_current_stream_usage_data'] = final_message.usage
                    except Exception as e_final:
                        st.warning(f"Could not get final message data: {e_final}")
                        st.session_state['_current_stream_usage_data'] = None
                else:
                    st.session_state['_current_stream_usage_data'] = None
                    
            except Exception as stream_error:
                st.error(f"Error during streaming: {stream_error}")
                yield f"Error during streaming: {stream_error}"
                st.session_state['_current_stream_usage_data'] = None

            st.session_state['_current_stream_full_text'] = full_response_text
        yield "" # Ensure stream finalization for st.write_stream
    except APIConnectionError as e:
        st.error(f"Anthropic API Connection Error: {e.__cause__}")
        yield f"Error: Connection failed. {e.__cause__}"
    except RateLimitError as e:
        st.error(f"Anthropic API Rate Limit Exceeded: {e}")
        yield f"Error: Rate limit exceeded. {e}"
    except APIStatusError as e:
        # Don't try to access streaming response content directly
        st.error(f"🚨 API Status Error Details:")
        st.error(f"Status Code: {e.status_code}")
        st.error(f"Original Exception: {str(e)}")
        
        # For streaming responses, don't try to access .text or .content
        if hasattr(e, 'response') and e.response:
            st.error(f"Response type: {type(e.response)}")
            # Only try to access headers, not content for streaming responses
            try:
                if hasattr(e.response, 'headers'):
                    headers = dict(e.response.headers)
                    st.error(f"Response headers: {headers}")
            except Exception as header_error:
                st.error(f"Could not access headers: {header_error}")
        
        yield f"Error: API Status {e.status_code}. See error details above."
    except APIError as e: # Catch more generic Anthropic errors
        st.error(f"Generic Anthropic API Error: {e}")
        yield f"Error: API Error. {e}"
    except Exception as e: # Catch any other unexpected errors
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e) # Show full traceback for unexpected errors
        yield f"Error: Unexpected issue. {e}"
    finally:
        st.session_state['streaming_in_progress'] = False
        # Reset stop_streaming flag *only if* this stream respected it or completed naturally.
        # If a new prompt caused the stop, that logic will handle resetting stop_streaming for the *new* stream.
        # This finally block executes for the *current* stream.
        if st.session_state['_stream_stopped_by_user_flag'] or not st.session_state.get("stop_streaming", False):
             st.session_state['stop_streaming'] = False


# --- Session State Initialization ---
# Ensure all session state variables are initialized here
default_session_state_values = {
    "messages": [],
    "session_total_usage": {},
    "session_total_cost": 0.0,
    "session_total_hypothetical_cost": 0.0,
    "streaming_in_progress": False,
    "stop_streaming": False,
    "_stream_stopped_by_user_flag": False,
    "_current_stream_usage_data": None,
    "_current_stream_full_text": "",
    "current_uploaded_files": [], # For files uploaded but not yet submitted with a prompt
    "prompt_submitted_this_run": False, # Helper to manage file uploader state
    "uploader_key_suffix": 0, # For resetting file_uploader by changing its key
}
for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- UI Rendering ---
st.title(f"🖼️ Multimodal Chatbot ({MODEL_NAME})")
st.caption("Supports text, image uploads, and PDF (summary/Q&A).")

# Placeholder for the stop button
stop_button_placeholder = st.empty()

# Display chat messages from history
for msg_idx, message_data in enumerate(st.session_state.messages):
    with st.chat_message(message_data["role"]):
        content = message_data["content"]
        if isinstance(content, str): # Old text-only format (backward compatibility)
            st.markdown(content)
        elif isinstance(content, list): # New unified API format
            for content_item in content:
                item_type = content_item.get("type")
                if item_type == "text":
                    st.markdown(content_item["text"])
                elif item_type == "image":
                    # Convert API format to display format
                    source = content_item.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        base64_data = source.get("data", "")
                        st.image(f"data:{media_type};base64,{base64_data}", caption="Image")
                elif item_type == "document":
                    st.caption("📄 PDF document")
        
        # Display usage and cost for assistant messages
        if message_data["role"] == "assistant":
            usage = message_data.get("usage")
            cost = message_data.get("cost", 0.0)
            stopped = message_data.get("stopped_by_user", False)

            if usage:
                details = []
                token_types_ordered = [
                    "input_tokens", "output_tokens", 
                    "cache_creation_input_tokens", "cache_read_input_tokens"
                ]
                for token_type in token_types_ordered:
                    count = 0
                    if isinstance(usage, dict):
                        count = usage.get(token_type, 0)
                    elif hasattr(usage, token_type):
                        count = getattr(usage, token_type, 0)
                    
                    if count > 0:
                        details.append(f"{token_type.replace('_', ' ').title()}: {count}")
                
                if cost > 0:
                    details.append(f"Cost: ${cost:.6f}")
                
                if details:
                    st.caption(", ".join(details))
            elif stopped:
                st.caption("[INFO] Streaming stopped. Token/cost info for this partial response might be unavailable.")

# Sidebar for Session Totals
with st.sidebar:
    st.header("📊 Session Statistics")
    st.subheader("Total Token Usage")
    total_usage_display = st.session_state.session_total_usage
    if not total_usage_display or all(val == 0 for val in total_usage_display.values()):
        st.info("No API calls yet or no tokens used.")
    else:
        # Display in a consistent order
        ordered_token_keys = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
        for token_type in ordered_token_keys:
            count = total_usage_display.get(token_type, 0)
            if count > 0: # Only display if there's usage for this type
                st.metric(
                    label=f"Total {token_type.replace('_', ' ').title()}",
                    value=f"{count:,}", # Format with comma for thousands
                )

    st.subheader("Total Estimated Cost")
    actual_session_cost = st.session_state.session_total_cost
    hypothetical_session_cost = st.session_state.session_total_hypothetical_cost
    
    cost_display_value = f"${actual_session_cost:.6f}"
    cost_label = "Session Cost"

    if hypothetical_session_cost > actual_session_cost and hypothetical_session_cost > 0:
        savings = hypothetical_session_cost - actual_session_cost
        savings_percentage = (savings / hypothetical_session_cost) * 100
        cost_label += f" (Saved {savings_percentage:.2f}%)"
        st.metric(label="Cost w/o Cache", value=f"${hypothetical_session_cost:.6f}", delta=f"-${savings:.6f} (Savings)", delta_color="inverse")
        st.metric(label=cost_label, value=cost_display_value)

    else:
        st.metric(label=cost_label, value=cost_display_value)


# Control Stop Button visibility
if st.session_state['streaming_in_progress']:
    with stop_button_placeholder.container():
        if st.button("🛑 Stop Generating", key="main_stop_button"):
            st.session_state['stop_streaming'] = True
            st.session_state['_stream_stopped_by_user_flag'] = True # Also set this internal flag
            st.info("Stop request processing... The current generation will halt shortly.")
            # No rerun here, the stream itself will stop and trigger a rerun via its processing flow
else:
    stop_button_placeholder.empty()


# --- File Uploader and Input Handling ---
# Place file uploader before chat input for better flow.

# Define the on_change callback for the file uploader
def on_uploader_change():
    current_key = f"file_uploader_widget_{st.session_state.get('uploader_key_suffix', 0)}"
    st.session_state.current_uploaded_files = st.session_state.get(current_key) or []

current_uploader_key = f"file_uploader_widget_{st.session_state.get('uploader_key_suffix', 0)}"

st.file_uploader(
    "📎 Attach Images (PNG, JPG, GIF, WEBP) or PDFs",
    type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
    accept_multiple_files=True,
    key=current_uploader_key, # Use dynamic key
    help="Upload images or PDF documents to discuss with the chatbot. PDFs will be summarized if possible.",
    on_change=on_uploader_change
    # Directly update current_uploaded_files when the uploader changes, ensuring it's a list.
)

# Manage current_uploaded_files based on widget interactions
# The on_change callback for file_uploader_widget now handles updating current_uploaded_files.
# We still need logic if prompt_submitted_this_run to clear it after submission (done later).

# Display previews of STAGED (but not yet submitted) files
if st.session_state['current_uploaded_files']:
    st.markdown("---")
    st.markdown("**Staged for next message:**")
    cols = st.columns(len(st.session_state['current_uploaded_files']))
    for idx, staged_file in enumerate(st.session_state['current_uploaded_files']):
        with cols[idx]:
            if staged_file.type.startswith("image/"):
                st.image(staged_file, caption=f"{staged_file.name[:20]}...", width=100)
            elif staged_file.type == "application/pdf":
                st.caption(f"📄 {staged_file.name[:20]}...")
    
    # Moved "Clear Attachments" button here
    if st.button("🗑️ Clear Staged Attachments", key="clear_files_button"):
        st.session_state['current_uploaded_files'] = []
        # Clear the file_uploader widget by resetting its value indirectly
        # Setting current_uploaded_files to [] and rerunning should make the uploader appear empty.
        st.rerun()
    st.markdown("---")

if prompt := st.chat_input("Ask about images/PDFs or send a message..."):
    st.session_state['prompt_submitted_this_run'] = True

    if st.session_state['streaming_in_progress']: # If a stream is active, stop it first
        st.session_state['stop_streaming'] = True
        st.info("Stopping current response to process new input...")
        # The old stream will eventually stop. We may need a slight delay or a more robust
        # mechanism to ensure it stops before the new one starts if it becomes an issue.
        # For now, Streamlit's execution model should handle this sequentially.

    # --- Construct User Message for API (store in API format from the start) ---
    user_message_content = [] # Single content array for both API and history

    # 1. Add files first
    if st.session_state['current_uploaded_files']:
        for uploaded_file_obj in st.session_state['current_uploaded_files']:
            file_bytes = uploaded_file_obj.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            media_type = uploaded_file_obj.type
            
            if not media_type or media_type == "application/octet-stream":
                 guessed_media_type, _ = mimetypes.guess_type(uploaded_file_obj.name)
                 if guessed_media_type:
                     media_type = guessed_media_type

            if media_type and media_type.startswith("image/"):
                if file_size_mb > 5:
                    st.error(f"Image '{uploaded_file_obj.name}' is too large ({file_size_mb:.1f}MB). Please use images smaller than 5MB.")
                    continue
                
                st.info(f"Processing image: {uploaded_file_obj.name} ({file_size_mb:.1f}MB, {media_type})")
                base64_image = base64.b64encode(file_bytes).decode("utf-8")
                user_message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    }
                })

            elif media_type == "application/pdf":
                if file_size_mb > 10:
                    st.error(f"PDF '{uploaded_file_obj.name}' is too large ({file_size_mb:.1f}MB). Please use PDFs smaller than 10MB.")
                    continue
                
                try:
                    st.info(f"Processing PDF: {uploaded_file_obj.name} ({file_size_mb:.1f}MB)")
                    base64_pdf_data = base64.b64encode(file_bytes).decode("utf-8")
                    user_message_content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf", 
                            "data": base64_pdf_data
                        }
                    })
                except Exception as e_pdf:
                    st.error(f"Could not process PDF {uploaded_file_obj.name}: {e_pdf}")

    # 2. Add text if provided
    if prompt:
        user_message_content.append({"type": "text", "text": prompt})

    # --- Add to History ---
    if not user_message_content:
        st.warning("Please enter a message or upload a file.")
        st.session_state['prompt_submitted_this_run'] = False
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_message_content})
    
    # --- Display User's Message Immediately ---
    with st.chat_message("user"):
        for content_item in user_message_content:
            if content_item.get("type") == "text":
                st.markdown(content_item["text"])
            elif content_item.get("type") == "image":
                # Convert API format back to display format
                source = content_item.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    base64_data = source.get("data", "")
                    st.image(f"data:{media_type};base64,{base64_data}", caption="Uploaded Image")
            elif content_item.get("type") == "document":
                st.caption("📄 PDF document uploaded")

    # Clear staged files
    st.session_state['current_uploaded_files'] = []
    st.session_state.uploader_key_suffix = st.session_state.get('uploader_key_suffix', 0) + 1

    # --- Prepare API Messages (direct copy since already in API format) ---
    api_messages_to_send = []
    for msg_data in st.session_state.messages:
        api_messages_to_send.append({
            "role": msg_data["role"], 
            "content": msg_data["content"]
        })

    # --- Apply Cache Control to Last Message's Last Content Element ---
    # Add cache_control to the last element of the last message's content before API call
    if api_messages_to_send:
        last_message = api_messages_to_send[-1]
        if "content" in last_message and isinstance(last_message["content"], list) and last_message["content"]:
            last_content_element = last_message["content"][-1]
            if isinstance(last_content_element, dict):
                # Create a copy to avoid modifying the original
                last_content_copy = last_content_element.copy()
                last_content_copy["cache_control"] = {"type": "ephemeral"}
                # Replace the last element with the copy that has cache_control
                last_message["content"][-1] = last_content_copy

    # --- Call API and Handle Response ---
    st.session_state['streaming_in_progress'] = True
    st.session_state['stop_streaming'] = False # Reset for the new stream
    st.session_state['_stream_stopped_by_user_flag'] = False
    st.session_state['_current_stream_full_text'] = ""
    st.session_state['_current_stream_usage_data'] = None

    # Debug: Check API key and basic setup
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY environment variable is not set!")
        st.session_state['streaming_in_progress'] = False
        st.stop()
    
    # Show basic debug info in an expander to keep UI clean
    with st.expander("🔍 Debug Info (click to expand)"):
        st.write(f"Model: {MODEL_NAME}")
        st.write(f"API Key present: {'Yes' if api_key else 'No'}")
        st.write(f"Number of messages to send: {len(api_messages_to_send)}")
        
        # Show the structure of messages being sent
        st.write("📋 Messages being sent to API:")
        for i, msg in enumerate(api_messages_to_send):
            st.write(f"Message {i+1} - Role: {msg['role']}")
            content = msg.get('content', [])
            if isinstance(content, list):
                for j, item in enumerate(content):
                    item_type = item.get('type', 'unknown')
                    if item_type == 'text':
                        text_preview = item.get('text', '')[:100] + '...' if len(item.get('text', '')) > 100 else item.get('text', '')
                        st.write(f"  Content {j+1}: text - {text_preview}")
                    elif item_type == 'image':
                        st.write(f"  Content {j+1}: image - {item.get('source', {}).get('media_type', 'unknown')} (base64 data)")
                    elif item_type == 'document':
                        st.write(f"  Content {j+1}: document - {item.get('source', {}).get('media_type', 'unknown')} (base64 data)")
                    else:
                        st.write(f"  Content {j+1}: {item_type}")
            else:
                st.write(f"  Content: {type(content)} - {str(content)[:100]}...")

    with st.chat_message("assistant"):
        # The stop button should be visible now due to streaming_in_progress=True and rerun.
        # Re-render the stop button placeholder ensure it's visible if it was previously empty.
        if st.session_state['streaming_in_progress']:
            with stop_button_placeholder.container():
                if st.button("🛑 Stop Generating", key="runtime_stop_button"): # different key if needed
                    st.session_state['stop_streaming'] = True
                    st.session_state['_stream_stopped_by_user_flag'] = True
                    st.info("Stop request processing...")
        else:
            stop_button_placeholder.empty()
            
        displayed_response_text = st.write_stream(
            get_anthropic_response_stream_with_usage(api_messages_to_send)
        )
        
        # Retrieve turn-specific results from session state (set by the generator)
        retrieved_usage_info = st.session_state['_current_stream_usage_data']
        retrieved_was_stopped_by_user = st.session_state['_stream_stopped_by_user_flag']
        
        # Clean up these session variables
        st.session_state['_current_stream_usage_data'] = None
        st.session_state['_stream_stopped_by_user_flag'] = False
        st.session_state['_current_stream_full_text'] = "" # Full text might be long, clear if not needed elsewhere

        final_assistant_message_content_text = displayed_response_text if displayed_response_text is not None else ""

        if retrieved_was_stopped_by_user:
            if not final_assistant_message_content_text.strip():
                final_assistant_message_content_text = "[INFO] Streaming was stopped by user before any content was generated."
            else:
                final_assistant_message_content_text = f"{final_assistant_message_content_text.rstrip()}\\n\\n[INFO] Streaming stopped by user."

        # Store assistant message in unified API format
        assistant_message_content = [{"type": "text", "text": final_assistant_message_content_text}]

        if final_assistant_message_content_text.strip() or retrieved_was_stopped_by_user:
            current_cost = 0.0
            current_turn_hypothetical_cost = 0.0 # Cost if no caching was used for this turn
            
            if retrieved_usage_info: # If we got usage data (API call was successful to some extent)
                model_costs_for_turn = get_model_costs(MODEL_NAME)

                # Calculate actual cost for the turn based on usage
                for token_type_api_name, cost_per_mil in model_costs_for_turn.items():
                    # Ensure we handle Pydantic model or dict for retrieved_usage_info
                    tokens_used_in_type = 0
                    if isinstance(retrieved_usage_info, dict):
                        tokens_used_in_type = retrieved_usage_info.get(token_type_api_name, 0)
                    elif hasattr(retrieved_usage_info, token_type_api_name):
                        tokens_used_in_type = getattr(retrieved_usage_info, token_type_api_name, 0)
                    
                    current_cost += (tokens_used_in_type / 1_000_000) * cost_per_mil

                    # Update session total usage
                    if tokens_used_in_type > 0:
                        st.session_state.session_total_usage[token_type_api_name] = \
                            st.session_state.session_total_usage.get(token_type_api_name, 0) + tokens_used_in_type
                
                st.session_state['session_total_cost'] += current_cost

                # Calculate hypothetical cost (primarily concerned with input/output if caching wasn't used)
                # This simplified hypothetical assumes cache reads would have been full input/output if not cached.
                # For cache_creation_input_tokens, they *are* input tokens, so their cost is part of hypothetical normal input.
                # For cache_read_input_tokens, if these were not read from cache, they would have been full output_tokens from a previous run.
                # This gets complex. A simpler hypothetical: "cost if all were new tokens".
                
                # Simpler hypothetical: cost of input tokens + output tokens at their standard rates.
                # This doesn't try to "reverse" caching benefits perfectly but gives a baseline.
                hypo_input_tokens = getattr(retrieved_usage_info, "input_tokens", 0) + \
                                    getattr(retrieved_usage_info, "cache_creation_input_tokens", 0) # cache creation is still input
                hypo_output_tokens = getattr(retrieved_usage_info, "output_tokens", 0)
                # If cache_read_input_tokens were used, what would they have cost if generated as output?
                # This is tricky. Let's use a simplified definition for hypothetical:
                # Cost of (input_tokens + cache_creation_input_tokens) as 'input'
                # Cost of (output_tokens) as 'output'
                # Cost of (cache_read_input_tokens) as if they were 'input_tokens' (as they replace a new query)
                
                current_turn_hypothetical_cost += (hypo_input_tokens / 1_000_000) * model_costs_for_turn.get("input_tokens", 0)
                current_turn_hypothetical_cost += (hypo_output_tokens / 1_000_000) * model_costs_for_turn.get("output_tokens", 0)
                
                # Add cost of cache_read_input_tokens as if they were fresh input_tokens for hypothetical scenario
                # This represents the cost saved by not having to re-send that part of the prompt.
                cache_read_tokens = getattr(retrieved_usage_info, "cache_read_input_tokens", 0)
                current_turn_hypothetical_cost += (cache_read_tokens / 1_000_000) * model_costs_for_turn.get("input_tokens", 0) # Cost them as input for hypo


                st.session_state['session_total_hypothetical_cost'] += current_turn_hypothetical_cost
            
            # Store assistant message
            usage_to_store = None
            if retrieved_usage_info:
                if hasattr(retrieved_usage_info, 'model_dump'): # Pydantic model
                    usage_to_store = retrieved_usage_info.model_dump()
                elif isinstance(retrieved_usage_info, dict): # Already a dict
                    usage_to_store = retrieved_usage_info

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message_content, # Store as list of dicts
                "usage": usage_to_store,
                "cost": current_cost,
                "stopped_by_user": retrieved_was_stopped_by_user
            })
            
            st.session_state['prompt_submitted_this_run'] = False # Reset flag for next input cycle
            st.rerun() # Rerun to update message display and clear input states
        else:
            # This case: no content from assistant AND not stopped by user (e.g. API error before stream)
            st.session_state['prompt_submitted_this_run'] = False # Reset flag
            # Check if an error message was already yielded by the stream and displayed
            # The `displayed_response_text` would contain it. If it's empty and not stopped, then show generic error.
            if not final_assistant_message_content_text.strip(): # no textual output from stream
                 # Check if last message already shows an error from the stream's error handling
                last_msg = st.session_state.messages[-1] if st.session_state.messages else {}
                is_last_msg_error = "Error:" in last_msg.get("content", [{}])[0].get("text","") if isinstance(last_msg.get("content"), list) and last_msg.get("content") else False

                if not is_last_msg_error:
                    st.error("Assistant did not return a message. An error might have occurred. Check terminal for logs.")

# Reset prompt submission flag if execution reaches here without a new prompt submission
# (e.g., after a rerun from "Clear Attachments" or "Stop Generating")
if not st.session_state.get("prompt_submitted_this_run", False) and not prompt:
    st.session_state['prompt_submitted_this_run'] = False
elif prompt: # If there was a prompt, it's handled above, ensure flag is reset before next cycle
    st.session_state['prompt_submitted_this_run'] = False 