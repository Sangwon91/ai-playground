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
DEFAULT_MODEL = "claude-sonnet-4-20250514" # Updated default to Sonnet for better multimodal
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

    # Apply cache control to the last assistant message if it exists and is suitable
    # The API expects cache_control on a specific content block within the message.
    # We'll target the last text block of the *second to last* message if it was an assistant.
    # This is because the *last* message is the user's current query.
    api_messages_with_cache_control = [msg.copy() for msg in api_messages_for_call] # Deep copy

    if len(api_messages_with_cache_control) >= 2: # Need at least user query + one previous message
        second_last_message = api_messages_with_cache_control[-2]
        if second_last_message.get("role") == "assistant":
            content_list = second_last_message.get("content")
            if isinstance(content_list, list):
                # Find the last text block to apply cache control
                for i in range(len(content_list) - 1, -1, -1):
                    if content_list[i].get("type") == "text":
                        # Ensure content_list[i] is a mutable copy if modifying
                        if not isinstance(content_list[i], dict): # Should be a dict
                            pass # Or log an error: malformed content
                        else:
                            # Create a copy of the content block to modify it
                            # This avoids modifying the original in st.session_state.messages
                            # The deep copy of api_messages_with_cache_control should handle this.
                            # Let's ensure we are modifying the copy:
                            
                            # If api_messages_with_cache_control was made from st.session_state.messages:
                            # original_content_block = second_last_message_original_ref["content"][i]
                            # api_messages_with_cache_control[-2]["content"][i] = original_content_block.copy()
                            # api_messages_with_cache_control[-2]["content"][i]["cache_control"] = {"type": "ephemeral"}
                            # Simpler: since `api_messages_with_cache_control` is already a copy of `api_messages_for_call`
                            # and `api_messages_for_call` should be a fresh build, direct modification here is okay.
                            # However, `content_list` itself might be a reference if not careful during api_messages_for_call build.
                            # To be safe, let's ensure the content block is copied if modifying history directly.
                            # For this function, api_messages_for_call is built fresh, so modifying its copy is fine.
                            
                            current_block_copy = content_list[i].copy()
                            current_block_copy["cache_control"] = {"type": "ephemeral"}
                            content_list[i] = current_block_copy # Replace with modified copy
                        break # Apply to only the last text block

    try:
        async with client.messages.stream(
            max_tokens=MAX_TOKENS_OUTPUT,
            messages=api_messages_with_cache_control, # Use the potentially modified list
            model=MODEL_NAME,
        ) as stream:
            async for text_chunk in stream.text_stream:
                if st.session_state.get("stop_streaming", False): # Check global stop signal
                    st.session_state['_stream_stopped_by_user_flag'] = True
                    break
                full_response_text += text_chunk
                yield text_chunk
            
            try:
                final_message = await stream.get_final_message()
                st.session_state['_current_stream_usage_data'] = final_message.usage
            except Exception as e_final:
                st.warning(f"Could not get final message data: {e_final}")
                st.session_state['_current_stream_usage_data'] = None # Ensure it's None on failure

            st.session_state['_current_stream_full_text'] = full_response_text
        yield "" # Ensure stream finalization for st.write_stream
    except APIConnectionError as e:
        st.error(f"Anthropic API Connection Error: {e.__cause__}")
        yield f"Error: Connection failed. {e.__cause__}"
    except RateLimitError as e:
        st.error(f"Anthropic API Rate Limit Exceeded: {e}")
        yield f"Error: Rate limit exceeded. {e}"
    except APIStatusError as e:
        st.error(f"Anthropic API Status Error (code {e.status_code}): {e.response}")
        yield f"Error: API Status {e.status_code}. {e.response}"
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
        content_to_display = message_data["content"]
        if isinstance(content_to_display, list): # New multimodal format
            for content_item in content_to_display:
                item_type = content_item.get("type")
                if item_type == "text":
                    st.markdown(content_item["text"])
                elif item_type == "image_url": # For displaying uploaded images
                    st.image(content_item["image_url"]["url"], caption=content_item.get("caption", "Image"))
                elif item_type == "tool_use" or item_type == "tool_result":
                     # Basic display for tool use, can be expanded
                    st.json(content_item)
            if message_data["role"] == "user" and not any(c.get("type") == "text" for c in content_to_display if isinstance(c,dict)):
                 # If user message had only non-text (e.g. only image), add a small note for clarity in history
                 st.caption("[User uploaded media without additional text]")

        elif isinstance(content_to_display, str): # Old format or pure text assistant response
            st.markdown(content_to_display)
        
        # Display usage and cost for assistant messages
        if message_data["role"] == "assistant":
            usage = message_data.get("usage")
            cost = message_data.get("cost", 0.0)
            stopped = message_data.get("stopped_by_user", False)

            if usage:
                details = []
                # Order for display
                token_types_ordered = [
                    "input_tokens", "output_tokens", 
                    "cache_creation_input_tokens", "cache_read_input_tokens"
                ]
                for token_type in token_types_ordered:
                    count = 0
                    # Usage can be a dict (from model_dump) or a Pydantic model
                    if isinstance(usage, dict):
                        count = usage.get(token_type, 0)
                    elif hasattr(usage, token_type): # Pydantic model object
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
# Using columns to place uploader and a potential "clear files" button side-by-side.
input_col1, input_col2 = st.columns([3,1])

with input_col1:
    uploaded_files_widget = st.file_uploader(
        "📎 Attach Images (PNG, JPG, GIF, WEBP) or PDFs",
        type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
        accept_multiple_files=True,
        key="file_uploader_widget", # Unique key for the widget
        help="Upload images or PDF documents to discuss with the chatbot. PDFs will be summarized if possible."
    )

with input_col2:
    if st.session_state['current_uploaded_files']: # Show clear button only if files are staged
        if st.button("Clear Attachments", key="clear_files_button"):
            st.session_state['current_uploaded_files'] = []
            # Clear the file uploader widget by resetting its key or value if necessary
            # Streamlit handles this by re-running. If current_uploaded_files is empty,
            # the uploader should reflect that on the next re-run.
            st.rerun() # Force rerun to update UI immediately

# Manage current_uploaded_files based on widget interactions
if uploaded_files_widget: # If new files are uploaded via the widget
    st.session_state['current_uploaded_files'] = uploaded_files_widget
elif not st.session_state['prompt_submitted_this_run'] and not uploaded_files_widget:
    # If no prompt was submitted (meaning it's a regular rerun) AND the widget is now empty,
    # it means the user manually cleared files from the widget.
    # So, update our session state list.
    # However, if `uploaded_files_widget` is None because it was just cleared by "Clear Attachments" button
    # and `st.session_state.current_uploaded_files` was already set to [], this is fine.
    pass # `current_uploaded_files` is already managed by the "Clear" button or if widget has new files.

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
    st.markdown("---")


if prompt := st.chat_input("Ask about images/PDFs or send a message..."):
    st.session_state['prompt_submitted_this_run'] = True

    if st.session_state['streaming_in_progress']: # If a stream is active, stop it first
        st.session_state['stop_streaming'] = True
        st.info("Stopping current response to process new input...")
        # The old stream will eventually stop. We may need a slight delay or a more robust
        # mechanism to ensure it stops before the new one starts if it becomes an issue.
        # For now, Streamlit's execution model should handle this sequentially.

    # --- Construct User Message for History and API ---
    user_message_content_for_history = [] # For display in Streamlit chat history
    user_message_content_for_api = []   # For sending to Anthropic API

    # 1. Process and Add Files from `st.session_state.current_uploaded_files` FIRST for API
    if st.session_state['current_uploaded_files']:
        for uploaded_file_obj in st.session_state['current_uploaded_files']:
            file_bytes = uploaded_file_obj.getvalue()
            media_type = uploaded_file_obj.type
            
            if not media_type or media_type == "application/octet-stream":
                 guessed_media_type, _ = mimetypes.guess_type(uploaded_file_obj.name)
                 if guessed_media_type:
                     media_type = guessed_media_type

            if media_type and media_type.startswith("image/"):
                base64_image = base64.b64encode(file_bytes).decode("utf-8")
                api_image_block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                    "cache_control": {"type": "ephemeral"}
                }
                user_message_content_for_api.append(api_image_block)
                
                # For history, order can be text then image for natural display
                # So, we'll add to history after text or handle order there.
                # For now, let's prepare history items and assemble later.

            elif media_type == "application/pdf":
                try:
                    base64_pdf_data = base64.b64encode(file_bytes).decode("utf-8")
                    api_pdf_block = {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf", 
                            "data": base64_pdf_data
                        },
                        "cache_control": {"type": "ephemeral"}
                    }
                    user_message_content_for_api.append(api_pdf_block)
                    st.info(f"PDF '{uploaded_file_obj.name}' prepared for sending to AI with input caching enabled.")
                except Exception as e_pdf:
                    st.error(f"Could not process PDF {uploaded_file_obj.name} for API: {e_pdf}")
                    # Add error to history as well
                    user_message_content_for_history.append({
                        "type": "text",
                        "text": f"[Error processing PDF '{uploaded_file_obj.name}' for API. It will not be sent.]"
                    })
            # Other file types are currently ignored for API but noted for history later

    # 2. Add Text part (if any) to API message list AFTER files
    if prompt:
        user_message_content_for_api.append({"type": "text", "text": prompt})

    # 3. Construct content for DISPLAY HISTORY (text first, then media)
    if prompt:
        user_message_content_for_history.append({"type": "text", "text": prompt})
    
    # Add successfully processed files and any errors to history
    if st.session_state['current_uploaded_files']:
        for uploaded_file_obj in st.session_state['current_uploaded_files']:
            media_type = uploaded_file_obj.type
            if not media_type or media_type == "application/octet-stream":
                 guessed_media_type, _ = mimetypes.guess_type(uploaded_file_obj.name)
                 if guessed_media_type:
                     media_type = guessed_media_type
            
            if media_type and media_type.startswith("image/"):
                # Check if this image was successfully added to the API list
                # This requires matching based on some unique aspect if direct object comparison is not feasible
                # For simplicity, we'll assume if it's an image, it was prepared for API.
                # A more robust check might involve comparing base64 data or names if necessary.
                was_added_to_api = any(
                    item.get("type") == "image" and item.get("source",{}).get("media_type") == media_type
                    # Add more specific checks if multiple images of same type can cause issues
                    for item in user_message_content_for_api
                )
                if was_added_to_api: # Simple check, assumes one image of a type for now or relies on order
                    file_bytes = uploaded_file_obj.getvalue()
                    base64_image = base64.b64encode(file_bytes).decode("utf-8")
                    user_message_content_for_history.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                        "caption": uploaded_file_obj.name
                    })
            elif media_type == "application/pdf":
                # Check if this PDF was successfully added to the API list
                pdf_api_item_exists = any(
                    item.get("type") == "document" and 
                    item.get("source", {}).get("media_type") == "application/pdf" and
                    # Potentially match by filename if storing original name in API block, or by data if feasible
                    # For now, presence of *any* PDF in API list implies this one if only one PDF is common.
                    # This check needs to be more robust if multiple PDFs are handled and one fails.
                    # Let's assume a simple check: if a PDF exists in API content, it was this one.
                    True # Simplified: if a PDF was to be processed, this is it.
                    for item in user_message_content_for_api if isinstance(item, dict) and item.get("type") == "document"
                )

                if pdf_api_item_exists:
                     # Check if an error was logged for *this specific file* during API prep
                    pdf_had_processing_error = False # Placeholder for more specific error tracking if needed
                    # For now, we rely on st.error having been called if there was an issue for this PDF.
                    # The st.info message for successful PDF preparation will be shown if no st.error for this file.
                    
                    # If no specific error logged for THIS pdf, assume success & show info.
                    # This logic can be improved by passing error status from API prep phase.
                    # A simpler way for history: if it's in API list, it was processed.
                    is_this_pdf_in_api_list = any(
                        item.get("type") == "document" and 
                        item.get("source", {}).get("media_type") == "application/pdf" and 
                        base64.b64encode(uploaded_file_obj.getvalue()).decode("utf-8") == item.get("source",{}).get("data")
                        for item in user_message_content_for_api if isinstance(item, dict)
                    )
                    if is_this_pdf_in_api_list:
                        user_message_content_for_history.append({
                            "type": "text", 
                            "text": f"📄 User uploaded PDF: {uploaded_file_obj.name} (Content sent to AI for analysis)."
                        })
                        st.info(f"PDF '{uploaded_file_obj.name}' was included in the API request.")
                    else:
                        # This implies it wasn't added to API list, likely due to an error caught in API prep loop
                        user_message_content_for_history.append({
                            "type": "text",
                            "text": f"[Error processing PDF '{uploaded_file_obj.name}'. It was not sent to the AI.]"
                         })
                # If processing failed above, an error like "Could not process PDF..." was already shown via st.error
                # and a history item for that error might have been added there too.
                # The goal here is to ensure history accurately reflects what happened.
            else: 
                user_message_content_for_history.append({
                    "type": "text",
                    "text": f"[User uploaded a file: {uploaded_file_obj.name} of type {media_type}. This type might not be processed by the AI.]"
                })
                st.warning(f"File '{uploaded_file_obj.name}' of type {media_type} may not be processed by the AI for the API call.")

    # Ensure there's at least one text block if images/docs were sent for API, as per Anthropic API requirement.
    # This is crucial if the user *only* uploads files and types no prompt.
    if any(item["type"] == "image" or item["type"] == "document" for item in user_message_content_for_api):
        if not any(item["type"] == "text" for item in user_message_content_for_api):
            # If only media, no text prompt from user, add a default text part to API content.
            user_message_content_for_api.append({"type": "text", "text": "Please analyze the following content."})
            # Also add to history for consistency, though user didn't type it
            if not prompt: # Only add this to history if there was genuinely no text prompt from user
                 user_message_content_for_history.insert(0, {"type": "text", "text": "[AI prompted to analyze uploaded content]"})


    # --- Add to History and Prepare for API ---
    if not user_message_content_for_history: # If only chat_input was an empty string and no files
        st.warning("Please enter a message or upload a file.")
        st.session_state['prompt_submitted_this_run'] = False # Reset flag
        st.stop() # Don't proceed

    st.session_state.messages.append({"role": "user", "content": user_message_content_for_history})
    
    # Clear the staged files now that they are part of the message
    st.session_state['current_uploaded_files'] = []
    # Potentially trigger a rerun here to clear the file_uploader widget if it doesn't clear automatically
    # This might be implicitly handled by the rerun after assistant's response.

    # --- Prepare API Message History ---
    # This needs to correctly format all messages, especially older ones.
    api_messages_to_send = []
    for msg_data in st.session_state.messages:
        api_role = msg_data["role"]
        api_content = None

        # Current turn's user message is already in API format
        if msg_data["content"] == user_message_content_for_history: #This is a bit fragile check
            # More robust: check if it's the last message and role is user
            if msg_data == st.session_state.messages[-1] and api_role == "user":
                 api_content = user_message_content_for_api # Use the specifically prepared API content
            # else: it's a historical user message, needs conversion if old format
        
        if not api_content: # If not the current user message, convert/prepare historical messages
            stored_content = msg_data["content"]
            if isinstance(stored_content, str): # Old text-only format
                api_content = [{"type": "text", "text": stored_content}]
            elif isinstance(stored_content, list): # New multimodal format (already stored for history)
                # Convert from history display format to API format if needed
                # e.g., "image_url" -> "image" with source, filter out display-only items
                temp_api_content = []
                for item in stored_content:
                    if item.get("type") == "text":
                        temp_api_content.append({"type": "text", "text": item["text"]})
                    # Assuming images in history are NOT already in API format.
                    # Images from user are processed above. Assistant responses are text.
                    # This part is mainly for assistant text messages stored in new list format.
                if temp_api_content:
                    api_content = temp_api_content
                elif api_role == "assistant": # If assistant message had no text (e.g. error display)
                     api_content = [{"type": "text", "text": "[Assistant message had no processable text content for API history]"}]


        if api_content: # Only add if content was successfully prepared
            api_messages_to_send.append({"role": api_role, "content": api_content})
        else:
            # Fallback for safety, should not happen with proper conversion
            st.warning(f"Skipping a message in history for API call due to formatting issue: {msg_data}")


    # --- Call API and Handle Response ---
    st.session_state['streaming_in_progress'] = True
    st.session_state['stop_streaming'] = False # Reset for the new stream
    st.session_state['_stream_stopped_by_user_flag'] = False
    st.session_state['_current_stream_full_text'] = ""
    st.session_state['_current_stream_usage_data'] = None

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

        # Prepare assistant message for history (always text for now)
        assistant_message_for_history = [{"type": "text", "text": final_assistant_message_content_text}]

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
                "content": assistant_message_for_history, # Store as list of dicts
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