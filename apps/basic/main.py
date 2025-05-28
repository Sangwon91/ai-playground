import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))


# --- Configuration & Constants ---
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
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
    "claude-3-5-haiku-20241022": {
        "input_tokens": 0.8,
        "output_tokens": 4,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 0.08,
    },
}

DEFAULT_MODEL_COSTS = COSTS_PER_MILLION_TOKENS.get(
    MODEL_NAME,  # Use current MODEL_NAME for default, not DEFAULT_MODEL
    {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
)

app = FastAPI()

# --- Anthropic Client Initialization ---
try:
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Anthropic client: {e}. Please ensure ANTHROPIC_API_KEY is correctly set.")
    # In a real app, you might want to prevent startup or handle this more gracefully
    client = None


# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="apps/basic/static"), name="static")

@app.get("/")
async def get():
    with open("apps/basic/static/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stop_streaming_flag = asyncio.Event()
    current_message_history = [] # Stores the full API message history for this connection

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")

            if action == "stop_streaming":
                stop_streaming_flag.set()
                continue

            if action == "send_message":
                user_message_content = payload.get("message")
                current_message_history.append({"role": "user", "content": user_message_content})

                full_response_text = ""
                current_stream_usage_data = None
                stream_stopped_by_user_flag = False
                stop_streaming_flag.clear() # Reset for the new stream

                # 1. Prepare a clean list of messages for the API call
                # This list will only contain {"role": ..., "content": ...}
                api_payload_messages = []
                for msg_hist_item in current_message_history:
                    api_payload_messages.append({
                        "role": msg_hist_item["role"],
                        "content": msg_hist_item["content"] 
                    })

                # 2. Apply cache control to the previous assistant message in this clean list, if applicable
                if len(api_payload_messages) >= 2 and api_payload_messages[-2]["role"] == "assistant":
                    assistant_message_to_modify = api_payload_messages[-2]
                    content_to_cache = assistant_message_to_modify["content"]

                    if isinstance(content_to_cache, str):
                        assistant_message_to_modify["content"] = [
                            {
                                "type": "text",
                                "text": content_to_cache,
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    elif isinstance(content_to_cache, list):
                        if content_to_cache and isinstance(content_to_cache[-1], dict):
                            content_to_cache[-1]["cache_control"] = {"type": "ephemeral"}
                
                try:
                    async with client.messages.stream(
                        max_tokens=2048,
                        messages=api_payload_messages, # Use the fully prepared list
                        model=MODEL_NAME,
                    ) as stream:
                        async for text_chunk in stream.text_stream:
                            if stop_streaming_flag.is_set():
                                stream_stopped_by_user_flag = True
                                await websocket.send_json({"type": "stop_ack"})
                                break
                            full_response_text += text_chunk
                            await websocket.send_json({"type": "text_chunk", "content": text_chunk})
                        
                        try:
                            final_message = await stream.get_final_message()
                            current_stream_usage_data = final_message.usage
                        except Exception as e_final:
                            print(f"Could not get final message data: {e_final}")
                            current_stream_usage_data = None

                    if stream_stopped_by_user_flag:
                         await websocket.send_json({"type": "stream_status", "status": "stopped_by_user", "full_text": full_response_text})
                    else:
                        await websocket.send_json({"type": "stream_status", "status": "completed", "full_text": full_response_text})


                    # Send usage data if available
                    if current_stream_usage_data:
                        current_cost = 0.0
                        current_turn_hypothetical_cost = 0.0
                        model_specific_costs = COSTS_PER_MILLION_TOKENS.get(MODEL_NAME, DEFAULT_MODEL_COSTS)

                        usage_dict = {
                            "input_tokens": current_stream_usage_data.input_tokens,
                            "output_tokens": current_stream_usage_data.output_tokens,
                            # Anthropic API may or may not return these depending on cache usage.
                            "cache_creation_input_tokens": getattr(current_stream_usage_data, 'cache_creation_input_tokens', 0),
                            "cache_read_input_tokens": getattr(current_stream_usage_data, 'cache_read_input_tokens', 0),
                        }

                        for token_type, cost_per_mil in model_specific_costs.items():
                            tokens_used = usage_dict.get(token_type, 0)
                            current_cost += (tokens_used / 1_000_000) * cost_per_mil
                        
                        current_turn_hypothetical_cost += (usage_dict.get("input_tokens",0) / 1_000_000) * model_specific_costs.get("input_tokens",0)
                        current_turn_hypothetical_cost += (usage_dict.get("output_tokens",0) / 1_000_000) * model_specific_costs.get("output_tokens",0)
                        # For hypothetical, cache creation is like a new input, cache read is like a new output (simplified model)
                        current_turn_hypothetical_cost += (usage_dict.get("cache_creation_input_tokens",0) / 1_000_000) * model_specific_costs.get("input_tokens",0)
                        current_turn_hypothetical_cost += (usage_dict.get("cache_read_input_tokens",0) / 1_000_000) * model_specific_costs.get("output_tokens",0)


                        await websocket.send_json({
                            "type": "usage_update",
                            "usage": usage_dict,
                            "cost": current_cost,
                            "hypothetical_cost": current_turn_hypothetical_cost,
                            "model_name": MODEL_NAME
                        })
                        current_message_history.append({"role": "assistant", "content": full_response_text, "usage": usage_dict, "cost": current_cost})
                    elif stream_stopped_by_user_flag:
                        # If stopped by user, we might not have usage, append message without it
                         current_message_history.append({"role": "assistant", "content": full_response_text, "stopped_by_user": True})
                    # If not stopped and no usage, it might be an error, already handled by error blocks.
                    
                except APIConnectionError as e:
                    error_msg = f"Anthropic API Connection Error: {e.__cause__}"
                    await websocket.send_json({"type": "error", "message": error_msg})
                    current_message_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
                except RateLimitError as e:
                    error_msg = f"Anthropic API Rate Limit Exceeded: {e}"
                    await websocket.send_json({"type": "error", "message": error_msg})
                    current_message_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
                except APIStatusError as e:
                    error_msg = f"Anthropic API Status Error (code {e.status_code}): {e.response}"
                    await websocket.send_json({"type": "error", "message": error_msg})
                    current_message_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
                except APIError as e:
                    error_msg = f"Generic Anthropic API Error: {e}"
                    await websocket.send_json({"type": "error", "message": error_msg})
                    current_message_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    await websocket.send_json({"type": "error", "message": error_msg})
                    current_message_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
                finally:
                    stop_streaming_flag.clear() # Ensure it's clear for any subsequent operations
                    await websocket.send_json({"type": "stream_end"}) # Signal client that stream processing is fully done


    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unhandled error in WebSocket: {e}")
        try: # Attempt to inform client if possible
            await websocket.send_json({"type": "error", "message": "An internal server error occurred."})
        except: # Client might already be gone
            pass
    finally:
        # Clean up resources if any specific to this connection
        print("WebSocket connection closed.")

# To run the app: uvicorn apps.basic.main:app --reload --port 8001
# Ensure your .env file is in the root of ai-playground-1 directory. 