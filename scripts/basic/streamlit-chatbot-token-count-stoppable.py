import os

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
DEFAULT_MODEL = "claude-3-5-sonnet-20240620" # Changed default to a more recent one
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

# Model-specific costs per million tokens
COSTS_PER_MILLION_TOKENS = {
    "claude-3-5-sonnet-20240620": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
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
    "claude-sonnet-4-20250514": { # Kept for consistency if user had this
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
    "claude-opus-4-20240229": { # Kept for consistency
        "input_tokens": 15.00,
        "output_tokens": 75.00,
        "cache_creation_input_tokens": 18.75,
        "cache_read_input_tokens": 1.50,
    },
     "claude-3-5-haiku-20241022": { # Kept for consistency
        "input_tokens": 0.8,
        "output_tokens": 4,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 0.08,
    },
}

DEFAULT_MODEL_COSTS = COSTS_PER_MILLION_TOKENS.get(
    DEFAULT_MODEL,
    {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
)

st.set_page_config(
    page_title=f"Stoppable Chatbot (Tokens & Cost) - {MODEL_NAME}", page_icon="🛑"
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


# --- Core API Interaction Function ---
async def get_anthropic_response_stream_with_usage(
    message_history_for_api: list,
):
    """
    Sends message history to Anthropic, enables caching on the last assistant message if applicable,
    yields response tokens. Usage data and full response text are stored in st.session_state.
    Allows for stopping the stream.
    """
    full_response_text = ""
    st.session_state._current_stream_usage_data = None
    st.session_state._current_stream_full_text = ""
    st.session_state.streaming_in_progress = True
    st.session_state._stream_stopped_by_user_flag = False # Internal flag

    api_messages_with_cache_control = []
    if message_history_for_api:
        api_messages_with_cache_control = [
            msg.copy() for msg in message_history_for_api
        ]
        if len(api_messages_with_cache_control) > 1:
            if isinstance(api_messages_with_cache_control[-1]["content"], list):
                if api_messages_with_cache_control[-1]["content"]:
                    api_messages_with_cache_control[-1]["content"][-1][
                        "cache_control"
                    ] = {"type": "ephemeral"}
            elif isinstance(
                api_messages_with_cache_control[-1]["content"], str
            ):
                api_messages_with_cache_control[-1]["content"] = [
                    {
                        "type": "text",
                        "text": api_messages_with_cache_control[-1]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
    else:
        api_messages_with_cache_control = []

    try:
        async with client.messages.stream(
            max_tokens=2048,
            messages=api_messages_with_cache_control
            if api_messages_with_cache_control
            else message_history_for_api,
            model=MODEL_NAME,
        ) as stream:
            async for text_chunk in stream.text_stream:
                if st.session_state.get("stop_streaming", False):
                    st.session_state._stream_stopped_by_user_flag = True
                    break  # Exit the loop if stop_streaming is True
                full_response_text += text_chunk
                yield text_chunk
            
            # Try to get final message even if stopped, might not have full usage if interrupted early.
            # The SDK might still provide partial usage or raise an error if the stream was abruptly closed.
            try:
                final_message = await stream.get_final_message()
                st.session_state._current_stream_usage_data = final_message.usage
            except Exception as e_final:
                # If getting final message fails after stopping, we might not have usage data.
                # This is okay, we'll handle None usage_data later.
                st.warning(f"Could not get final message data after stream stop/completion: {e_final}")
                st.session_state._current_stream_usage_data = None

            st.session_state._current_stream_full_text = full_response_text
        yield ""  # Ensure stream finalization for st.write_stream
    except APIConnectionError as e:
        st.error(f"Anthropic API Connection Error: {e.__cause__}")
        st.session_state._current_stream_full_text = full_response_text
        yield f"Error: Connection failed. {e.__cause__}"
    except RateLimitError as e:
        st.error(f"Anthropic API Rate Limit Exceeded: {e}")
        st.session_state._current_stream_full_text = full_response_text
        yield f"Error: Rate limit exceeded. {e}"
    except APIStatusError as e:
        st.error(
            f"Anthropic API Status Error (code {e.status_code}): {e.response}"
        )
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
    finally:
        st.session_state.streaming_in_progress = False
        st.session_state.stop_streaming = False # Reset for next turn


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_total_usage" not in st.session_state:
    st.session_state.session_total_usage = {}
if "session_total_cost" not in st.session_state:
    st.session_state.session_total_cost = 0.0
if "session_total_hypothetical_cost" not in st.session_state:
    st.session_state.session_total_hypothetical_cost = 0.0
if "streaming_in_progress" not in st.session_state: # To control UI elements like stop button
    st.session_state.streaming_in_progress = False
if "stop_streaming" not in st.session_state: # Signal to stop the current stream
    st.session_state.stop_streaming = False
if "_stream_stopped_by_user_flag" not in st.session_state: # To append info message
    st.session_state._stream_stopped_by_user_flag = False


# --- UI Rendering ---
st.title("🤖 Stoppable Chatbot with Token Counting & Cost")
st.caption(f"Using model: {MODEL_NAME}")

# Placeholder for the stop button - Define it before messages and input for stable positioning
# This placeholder will be populated when streaming_in_progress is true.
stop_button_placeholder = st.empty()

# Display chat messages from history
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
        ): # Check assistant role first
            usage = message.get("usage") # Use .get for safer access
            cost = message.get("cost", 0.0)
            stopped = message.get("stopped_by_user", False)

            if usage: # If usage data exists, display it
                details = []
                token_types_ordered = [
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ]
                for token_type in token_types_ordered:
                    count = 0
                    if isinstance(usage, dict) and token_type in usage:
                        count = usage[token_type]
                    elif hasattr(usage, token_type): # Handles Pydantic model
                        count = getattr(usage, token_type)
                    
                    if count > 0:
                        details.append(
                            f"{token_type.replace('_', ' ').title()}: {count}"
                        )

                if cost > 0:
                    details.append(f"Cost: ${cost:.6f}")

                if details:
                    st.caption(", ".join(details))
            elif stopped: # No usage data, but we know it was stopped by the user
                st.caption("[INFO] Streaming was stopped by the user. Full token usage for this partial response is unavailable.")
            # If neither usage nor stopped, no specific usage/token caption is added for this message.


# Sidebar for Session Totals
with st.sidebar:
    st.header("Session Statistics")
    st.subheader("Total Token Usage")
    total_usage_display = st.session_state.session_total_usage
    if not total_usage_display:
        st.info("No API calls yet in this session.")
    for token_type, count in total_usage_display.items():
        if count > 0:
            st.metric(
                label=f"Total {token_type.replace('_', ' ').title()}",
                value=count,
            )

    st.subheader("Total Estimated Cost")
    actual_session_cost = st.session_state.session_total_cost
    hypothetical_session_cost = st.session_state.session_total_hypothetical_cost
    cost_display_value = f"${actual_session_cost:.6f}"
    label = "Session Cost"
    if (
        hypothetical_session_cost > 0
        and hypothetical_session_cost > actual_session_cost
    ):
        savings = hypothetical_session_cost - actual_session_cost
        if hypothetical_session_cost > 0: # Avoid division by zero
            savings_percentage = (savings / hypothetical_session_cost) * 100
            label += f" (Saved {savings_percentage:.2f}%)"
        else:
            label += " (Savings N/A)"


    st.metric(label=label, value=cost_display_value)

# Control Stop Button visibility based on streaming state.
if st.session_state.get("streaming_in_progress", False):
    with stop_button_placeholder.container():
        if st.button("Stop Generating", key="main_stop_button"):
            st.session_state.stop_streaming = True
            st.info("Stop request processing...") 
else:
    stop_button_placeholder.empty() # Clear button if not streaming


# Handle user input
if prompt := st.chat_input("What would you like to ask?"):
    if st.session_state.get("streaming_in_progress", False): # Check before modifying state for new prompt
        st.session_state.stop_streaming = True # Signal to stop current stream
        st.info("Stopping current response to process new input...")
        # The old stream will see this flag and stop. Its `finally` block will set streaming_in_progress=False.
        # We proceed to set up for the new stream.

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Initialize states for the NEW stream about to start
    st.session_state.streaming_in_progress = True
    st.session_state.stop_streaming = False # Critical: ensure new stream doesn't immediately stop
    st.session_state._stream_stopped_by_user_flag = False # Reset this flag for the new stream
    st.session_state._current_stream_full_text = "" # Initialize for the new stream
    st.session_state._current_stream_usage_data = None # Initialize for the new stream

    with st.chat_message("assistant"):
        # streaming_in_progress is True now. The button logic above this `if prompt` block
        # will ensure the button is displayed on the rerun triggered by st.chat_input processing or st.rerun later.
        
        displayed_response_text = st.write_stream(
            get_anthropic_response_stream_with_usage(api_messages)
        )
        
        # After the stream (completed or stopped), streaming_in_progress is set to False by the generator's finally block.
        # The button will be removed on the st.rerun() call below.

        # Retrieve turn-specific results from session state (set by the generator)
        retrieved_usage_info = st.session_state.get('_current_stream_usage_data')
        retrieved_was_stopped_by_user = st.session_state.get('_stream_stopped_by_user_flag', False)
        
        # Clean up these session variables now that we have their values for this turn
        st.session_state.pop('_current_stream_usage_data', None)
        st.session_state.pop('_stream_stopped_by_user_flag', None)
        st.session_state.pop('_current_stream_full_text', None) # Clean up the generator's internal accumulator too

        # Use the text that was actually displayed by st.write_stream
        # Ensure it's a string, even if None was returned by st.write_stream (though unlikely with yield "")
        final_assistant_message_content = displayed_response_text if displayed_response_text is not None else ""

        if retrieved_was_stopped_by_user:
            if not final_assistant_message_content.strip(): # If stopped and message is empty
                final_assistant_message_content = "[INFO] Streaming was stopped by user before any content was generated.]"
            else:
                # Append to the existing content, ensuring no double newlines if original text ended with one.
                final_assistant_message_content = f"{final_assistant_message_content.rstrip()}\n\n[INFO] Streaming stopped by user."

        # Add assistant message to history if there's content or if it was stopped (even if no actual text content from LLM)
        if final_assistant_message_content.strip() or retrieved_was_stopped_by_user:
            current_cost = 0.0
            current_turn_hypothetical_cost = 0.0
            
            is_successful_api_response = retrieved_usage_info is not None 

            if is_successful_api_response:
                model_specific_costs = COSTS_PER_MILLION_TOKENS.get(
                    MODEL_NAME, DEFAULT_MODEL_COSTS
                )

                for token_type, cost_per_mil in model_specific_costs.items():
                    tokens_used = getattr(retrieved_usage_info, token_type, 0)
                    current_cost += (tokens_used / 1_000_000) * cost_per_mil

                current_turn_hypothetical_cost += (
                    getattr(retrieved_usage_info, "input_tokens", 0) / 1_000_000
                ) * model_specific_costs.get("input_tokens", 0)
                current_turn_hypothetical_cost += (
                    getattr(retrieved_usage_info, "output_tokens", 0) / 1_000_000
                ) * model_specific_costs.get("output_tokens", 0)
                current_turn_hypothetical_cost += (
                    getattr(retrieved_usage_info, "cache_creation_input_tokens", 0) / 1_000_000
                ) * model_specific_costs.get("input_tokens",0)
                current_turn_hypothetical_cost += (
                    getattr(retrieved_usage_info, "cache_read_input_tokens", 0) / 1_000_000
                ) * model_specific_costs.get("output_tokens",0)

                for token_type_key in model_specific_costs.keys():
                    tokens_in_turn = getattr(retrieved_usage_info, token_type_key, 0)
                    if tokens_in_turn > 0:
                        st.session_state.session_total_usage[token_type_key] = (
                            st.session_state.session_total_usage.get(
                                token_type_key, 0
                            )
                            + tokens_in_turn
                        )
                st.session_state.session_total_cost += current_cost
                st.session_state.session_total_hypothetical_cost += (
                    current_turn_hypothetical_cost
                )
            
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_assistant_message_content,
                    "usage": retrieved_usage_info.model_dump() if retrieved_usage_info and hasattr(retrieved_usage_info, 'model_dump') else (retrieved_usage_info if isinstance(retrieved_usage_info, dict) else None),
                    "cost": current_cost,
                    "stopped_by_user": retrieved_was_stopped_by_user
                }
            )
            st.rerun()
        else:
            # This case means the generator yielded nothing and was not stopped by user explicitly.
            if not any(
                msg["role"] == "assistant"
                and (msg["content"].startswith("Error:") or (msg.get("usage") is None and not msg.get("stopped_by_user")))
                for msg in st.session_state.messages[-2:] # check last two for safety
            ):
                st.error(
                    "Assistant did not return a message or an error. Please check logs or try again."
                )

# Redundant check for stop button visibility, main logic is above `if prompt`
# This can be removed as the primary logic is now always active based on `streaming_in_progress`
# if st.session_state.get("streaming_in_progress", False):
# with stop_button_placeholder.container():
# if st.button("Stop Generating", key="stop_button_visible_on_rerun"):
# st.session_state.stop_streaming = True
# st.info("Stop request received. Finishing current chunk...")
# elif not st.session_state.get("streaming_in_progress", False):
# stop_button_placeholder.empty() 