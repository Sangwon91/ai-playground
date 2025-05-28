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
    MODEL_NAME,
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
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Anthropic client: {e}. Please ensure ANTHROPIC_API_KEY is correctly set.")
    anthropic_client = None


# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="apps/basic/static"), name="static")

@app.get("/")
async def get_index():
    with open("apps/basic/static/index.html") as f:
        return HTMLResponse(f.read())

async def _execute_anthropic_stream(
    websocket: WebSocket,
    api_messages_for_call: list,
    stop_event: asyncio.Event,
    model_name: str,
    costs_per_million: dict,
    default_costs: dict
):
    full_response_text = ""
    current_stream_usage_data = None
    stream_stopped_by_user_flag = False
    error_msg_val = None # Initialize to avoid issues if checked before assignment in an edge case

    try:
        async with anthropic_client.messages.stream(
            max_tokens=2048,
            messages=api_messages_for_call,
            model=model_name,
        ) as stream:
            print(f"[DEBUG] Stream {stream_id if (stream_id := id(stream)) else 'N/A'} started. Listening for stop_event {id(stop_event)}.")
            async for text_chunk in stream.text_stream:
                if stop_event.is_set():
                    print(f"[DEBUG] Stop event {id(stop_event)} is SET in stream {stream_id}. Breaking loop.")
                    stream_stopped_by_user_flag = True
                    break
                full_response_text += text_chunk
                await websocket.send_json({"type": "text_chunk", "content": text_chunk})
            
            try:
                final_message = await stream.get_final_message()
                current_stream_usage_data = final_message.usage
            except Exception as e_final:
                print(f"Could not get final message data after stream: {e_final}")
                current_stream_usage_data = None

        status_type = "stopped_by_user" if stream_stopped_by_user_flag else "completed"
        await websocket.send_json({"type": "stream_status", "status": status_type, "full_text": full_response_text})

        if current_stream_usage_data:
            current_cost = 0.0
            current_turn_hypothetical_cost = 0.0
            model_specific_costs = costs_per_million.get(model_name, default_costs)
            usage_dict = {
                "input_tokens": current_stream_usage_data.input_tokens,
                "output_tokens": current_stream_usage_data.output_tokens,
                "cache_creation_input_tokens": getattr(current_stream_usage_data, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(current_stream_usage_data, 'cache_read_input_tokens', 0),
            }
            for token_type_loop, cost_per_mil in model_specific_costs.items(): # Renamed token_type to avoid conflict
                tokens_used = usage_dict.get(token_type_loop, 0)
                current_cost += (tokens_used / 1_000_000) * cost_per_mil
            
            current_turn_hypothetical_cost += (usage_dict.get("input_tokens",0) / 1_000_000) * model_specific_costs.get("input_tokens",0)
            current_turn_hypothetical_cost += (usage_dict.get("output_tokens",0) / 1_000_000) * model_specific_costs.get("output_tokens",0)
            current_turn_hypothetical_cost += (usage_dict.get("cache_creation_input_tokens",0) / 1_000_000) * model_specific_costs.get("input_tokens",0)
            current_turn_hypothetical_cost += (usage_dict.get("cache_read_input_tokens",0) / 1_000_000) * model_specific_costs.get("output_tokens",0)

            await websocket.send_json({
                "type": "usage_update",
                "usage": usage_dict,
                "cost": current_cost,
                "hypothetical_cost": current_turn_hypothetical_cost,
                "model_name": model_name
            })
            return {"role": "assistant", "content": full_response_text, "usage": usage_dict, "cost": current_cost}
        elif stream_stopped_by_user_flag:
            return {"role": "assistant", "content": full_response_text, "stopped_by_user": True}
        
        # If try completes without usage data and not stopped (e.g. empty response from API but no error)
        return {"role": "assistant", "content": full_response_text} # Or None, depending on desired history for empty valid response

    except asyncio.CancelledError:
        print("Stream task was cancelled.")
        try:
            await websocket.send_json({
                "type": "stream_status", 
                "status": "stopped_by_user", 
                "full_text": full_response_text + " [Cancelled by Server]"
            })
        except Exception as send_err:
            print(f"Error sending cancellation status to client: {send_err}")
        return {"role": "assistant", "content": full_response_text + " [Stream Cancelled]", "stopped_by_user": True}
    
    except (APIConnectionError, RateLimitError, APIStatusError, APIError, Exception) as e:
        # Determine error_msg_val based on type of e
        if isinstance(e, APIConnectionError): error_msg_val = f"Anthropic API Connection Error: {e.__cause__}"
        elif isinstance(e, RateLimitError): error_msg_val = f"Anthropic API Rate Limit Exceeded: {e}"
        elif isinstance(e, APIStatusError): error_msg_val = f"Anthropic API Status Error (code {e.status_code}): {e.response}"
        elif isinstance(e, APIError): error_msg_val = f"Generic Anthropic API Error: {e}"
        else: error_msg_val = f"An unexpected error occurred in stream execution: {e}" # General Exception

        print(f"Error in stream: {error_msg_val}")
        try:
            await websocket.send_json({"type": "error", "message": error_msg_val})
        except Exception as send_err:
            print(f"Error sending error message to client: {send_err}")
        return {"role": "assistant", "content": f"Error: {error_msg_val}"} 

    else:
        # This block executes if the TRY block completed WITHOUT any exceptions AND no return occurred within TRY.
        # Given current logic, TRY block always returns on success/stop/empty. So this else might be less critical
        # but is good for completeness of the try-except-else-finally structure.
        print("Stream processing path in 'else' reached, implies try completed with no exceptions and no explicit return from try.")
        return None # Explicitly return None if try completed but didn't yield specific data for history.

    finally:
        # This ALWAYS executes after try/except/else, before the function actually returns.
        try:
            await websocket.send_json({"type": "stream_end"})
        except Exception as send_err:
            print(f"Error sending stream_end to client: {send_err}")

async def _handle_stream_completion(task: asyncio.Task, message_history: list):
    """Helper to await task and handle its result or exceptions, adding to history."""
    try:
        assistant_response_data = await task
        if assistant_response_data:
            message_history.append(assistant_response_data)
    except asyncio.CancelledError:
        print("[DEBUG] Stream task was cancelled (handled in _handle_stream_completion).")
        # The task itself should have added a [Cancelled] message to history
    except Exception as e:
        print(f"[DEBUG] Error awaiting stream task in _handle_stream_completion: {e}")
        message_history.append({"role": "assistant", "content": f"Error: Unhandled issue awaiting stream task."})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    current_message_history = [] 
    # These need to be managed carefully across iterations of the while loop
    active_stream_task: asyncio.Task | None = None
    active_stop_event: asyncio.Event | None = None

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")

            if action == "stop_streaming":
                print(f"[DEBUG] Received 'stop_streaming' action from client.")
                if active_stream_task and not active_stream_task.done() and active_stop_event:
                    print(f"[DEBUG] Active stream task found. Setting active_stop_event {id(active_stop_event)}.")
                    active_stop_event.set()
                    await websocket.send_json({"type": "stop_ack", "message": "Stop signal sent to current stream."})
                else:
                    print("[DEBUG] No active stream task to stop, or active_stop_event is None/task done.")
                    await websocket.send_json({"type": "info", "message": "No active stream to stop."})
                continue

            if action == "send_message":
                print(f"[DEBUG] Received 'send_message' action.")
                
                # If there's an existing stream task, signal it to stop and wait for it.
                if active_stream_task and not active_stream_task.done():
                    if active_stop_event:
                        print(f"[DEBUG] New message while old stream running. Signaling old stream (event {id(active_stop_event)}) to stop.")
                        active_stop_event.set()
                    else:
                        print("[DEBUG] Old stream running but no active_stop_event. Cancelling directly.")
                        active_stream_task.cancel() # Fallback if event was somehow lost
                    
                    # Wait for the old task to actually complete/cancel
                    print(f"[DEBUG] Waiting for old stream task {id(active_stream_task)} to finish...")
                    await _handle_stream_completion(active_stream_task, current_message_history)
                    print(f"[DEBUG] Old stream task {id(active_stream_task)} finished.")
                
                active_stream_task = None # Clear previous task reference
                active_stop_event = None  # Clear previous event reference

                current_message_history.append({"role": "user", "content": payload.get("message")})
                
                api_payload_messages = [] # Prepare messages for API (as before)
                for msg_hist_item in current_message_history:
                    api_payload_messages.append({
                        "role": msg_hist_item["role"],
                        "content": msg_hist_item["content"] 
                    })
                if len(api_payload_messages) >= 2 and api_payload_messages[-2]["role"] == "assistant":
                    assistant_message_to_modify = api_payload_messages[-2]
                    content_to_cache = assistant_message_to_modify["content"]
                    if isinstance(content_to_cache, str):
                        assistant_message_to_modify["content"] = [{"type": "text", "text": content_to_cache, "cache_control": {"type": "ephemeral"}}]
                    elif isinstance(content_to_cache, list):
                        if content_to_cache and isinstance(content_to_cache[-1], dict):
                            content_to_cache[-1]["cache_control"] = {"type": "ephemeral"}
                
                active_stop_event = asyncio.Event() # Create a NEW event for THIS stream
                print(f"[DEBUG] Created new active_stop_event {id(active_stop_event)} for new stream.")
                
                # Launch the new stream task but DO NOT await it here directly.
                active_stream_task = asyncio.create_task(
                    _execute_anthropic_stream(
                        websocket, 
                        api_payload_messages, 
                        active_stop_event, 
                        MODEL_NAME, 
                        COSTS_PER_MILLION_TOKENS, 
                        DEFAULT_MODEL_COSTS
                    )
                )
                # Instead of awaiting here, we can optionally add a done callback to handle its completion/errors for history
                # For simplicity now, _handle_stream_completion will be called if a *new* message arrives or on disconnect.
                # A more robust solution might involve a separate task monitoring active_stream_task.
                print(f"[DEBUG] Launched new stream task {id(active_stream_task)}. Main loop continues.")

    except WebSocketDisconnect:
        print("[DEBUG] Client disconnected")
        if active_stream_task and not active_stream_task.done():
            if active_stop_event: active_stop_event.set()
            active_stream_task.cancel()
            print("[DEBUG] Disconnected: Active stream task cancelled.")
            # Await its completion to ensure cleanup messages are potentially processed by it
            await _handle_stream_completion(active_stream_task, current_message_history)
            print("[DEBUG] Awaited cancelled task on disconnect.")

    except Exception as e:
        print(f"[DEBUG] Unhandled error in WebSocket main loop: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "An internal server error occurred."})
        except: pass
    finally:
        # Final cleanup if a task is still somehow pending (should be handled by disconnect normally)
        if active_stream_task and not active_stream_task.done():
            print("[DEBUG] Final cleanup: Cancelling a lingering active stream task.")
            if active_stop_event and not active_stop_event.is_set(): active_stop_event.set()
            active_stream_task.cancel()
            # Don't necessarily need to await here as connection is closing
        print("[DEBUG] WebSocket connection closed.")

# To run the app: uvicorn apps.basic.main:app --reload --port 8001
# Ensure your .env file is in the root of ai-playground-1 directory. 