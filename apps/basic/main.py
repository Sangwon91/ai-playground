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
            async for text_chunk in stream.text_stream:
                if stop_event.is_set():
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    current_message_history = [] 
    current_stream_task: asyncio.Task | None = None
    current_stop_event: asyncio.Event | None = None

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")

            if action == "stop_streaming":
                if current_stream_task and not current_stream_task.done() and current_stop_event:
                    current_stop_event.set()
                    await websocket.send_json({"type": "stop_ack", "message": "Stop signal sent to current stream."})
                else:
                    await websocket.send_json({"type": "info", "message": "No active stream to stop."})
                continue

            if action == "send_message":
                if current_stream_task and not current_stream_task.done():
                    if current_stop_event:
                        print("New message received while old stream running. Signaling old stream to stop.")
                        current_stop_event.set()
                    try:
                        # Give the old task a moment to stop gracefully
                        await asyncio.wait_for(current_stream_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        print("Old stream did not stop in time, cancelling.")
                        current_stream_task.cancel()
                    except Exception as e:
                        print(f"Error while waiting for old stream to stop: {e}")
                    # Wait for task to actually finish after cancellation attempt
                    if not current_stream_task.done(): # Ensure its done before nullifying
                         try: await current_stream_task 
                         except asyncio.CancelledError: print("Old task was indeed cancelled.")
                         except Exception: pass # Other exceptions would have been handled inside task

                current_message_history.append({"role": "user", "content": payload.get("message")})
                
                api_payload_messages = []
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
                
                current_stop_event = asyncio.Event()
                current_stream_task = asyncio.create_task(
                    _execute_anthropic_stream(
                        websocket, 
                        api_payload_messages, 
                        current_stop_event, 
                        MODEL_NAME, 
                        COSTS_PER_MILLION_TOKENS, 
                        DEFAULT_MODEL_COSTS
                    )
                )
                
                try:
                    # Wait for the task to complete and get its result (assistant message for history)
                    assistant_response_data = await current_stream_task 
                    if assistant_response_data:
                        current_message_history.append(assistant_response_data)
                except asyncio.CancelledError:
                    print("Stream task was cancelled externally (likely by new message or disconnect).")
                    # Message history for cancelled part handled by _execute_anthropic_stream's CancelledError block
                except Exception as e:
                    print(f"Error awaiting stream task: {e}")
                    current_message_history.append({"role": "assistant", "content": f"Error: Unhandled issue with stream execution."})


    except WebSocketDisconnect:
        print("Client disconnected")
        if current_stream_task and not current_stream_task.done():
            if current_stop_event: current_stop_event.set() # Signal before cancel
            current_stream_task.cancel()
            print("Disconnected: Active stream task cancelled.")
    except Exception as e:
        print(f"Unhandled error in WebSocket main loop: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "An internal server error occurred."})
        except: pass # Client might be gone
    finally:
        if current_stream_task and not current_stream_task.done(): # Final cleanup if somehow missed
            current_stream_task.cancel()
            try:
                await current_stream_task 
            except:
                pass
        print("WebSocket connection closed.")

# To run the app: uvicorn apps.basic.main:app --reload --port 8001
# Ensure your .env file is in the root of ai-playground-1 directory. 