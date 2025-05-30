import os
import base64
import mimetypes
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import threading
import nest_asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MCP_Chatbot')

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

import streamlit as st
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from dotenv import load_dotenv

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
MAX_TOKENS_OUTPUT = 4096

# Model costs
COSTS_PER_MILLION_TOKENS = {
    "claude-3-5-sonnet-20241022": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
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

# --- MCP Configuration Classes ---
@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPServerConfig':
        return cls(
            name=data.get("name", ""),
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env"),
            enabled=data.get("enabled", True)
        )

# --- MCP Connection Manager ---
class MCPConnectionManager:
    """Manages MCP server configurations and creates connections on demand"""
    
    def __init__(self):
        self.configs = {}
        self.lock = threading.Lock()
    
    def add_config(self, server_name: str, config: MCPServerConfig):
        """Add or update server configuration"""
        with self.lock:
            self.configs[server_name] = config
    
    def remove_config(self, server_name: str):
        """Remove server configuration"""
        with self.lock:
            self.configs.pop(server_name, None)
    
    def get_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get server configuration"""
        with self.lock:
            return self.configs.get(server_name)
    
    def get_all_configs(self) -> Dict[str, MCPServerConfig]:
        """Get all server configurations"""
        with self.lock:
            return self.configs.copy()
    
    async def create_temporary_connection(self, server_name: str) -> Optional[Tuple[ClientSession, AsyncExitStack, List[Any]]]:
        """Create a temporary connection for a single request"""
        config = self.get_config(server_name)
        if not config or not config.enabled:
            return None
        
        logger.info(f"Creating temporary connection for {server_name}")
        exit_stack = AsyncExitStack()
        
        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )
            
            # Start MCP server
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio_read, stdio_write = stdio_transport
            
            # Create session
            session = await exit_stack.enter_async_context(
                ClientSession(stdio_read, stdio_write)
            )
            
            # Initialize session
            await session.initialize()
            
            # Get tools
            tools_response = await session.list_tools()
            tools = tools_response.tools if hasattr(tools_response, 'tools') else []
            
            logger.info(f"Temporary connection created for {server_name} with {len(tools)} tools")
            return session, exit_stack, tools
            
        except Exception as e:
            logger.error(f"Failed to create temporary connection for {server_name}: {e}")
            await exit_stack.aclose()
            return None

# --- Page Configuration ---
st.set_page_config(
    page_title=f"MCP Multimodal Chatbot - {MODEL_NAME}", 
    page_icon="🔧",
    layout="wide"
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
    """Safely retrieves costs for a given model"""
    return COSTS_PER_MILLION_TOKENS.get(model_name_to_check, DEFAULT_MODEL_COSTS)

def run_async(coro):
    """Run async function in a way compatible with Streamlit"""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a future and run the coroutine in the current loop
            import nest_asyncio
            nest_asyncio.apply()
            task = loop.create_task(coro)
            # Use asyncio.run_coroutine_threadsafe for thread safety
            return loop.run_until_complete(task)
        else:
            # If no loop is running, create one
            return asyncio.run(coro)
    except RuntimeError:
        # Fallback: create new event loop
        return asyncio.run(coro)

# --- MCP Functions ---
async def call_mcp_tool_with_new_connection(manager: MCPConnectionManager, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool on an MCP server using a fresh connection"""
    logger.info(f"Calling tool {tool_name} on server {server_name}")
    
    # Create temporary connection
    connection_info = await manager.create_temporary_connection(server_name)
    if not connection_info:
        error_msg = f"Failed to create connection to server '{server_name}'"
        logger.error(error_msg)
        return error_msg
    
    session, exit_stack, _ = connection_info
    
    try:
        # Call the tool
        logger.info(f"Executing tool call: {tool_name} with args: {arguments}")
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=30.0
        )
        
        # Extract content from result
        if hasattr(result, 'content'):
            content_list = result.content
            text_parts = []
            for content_item in content_list:
                if hasattr(content_item, 'type') and content_item.type == 'text':
                    if hasattr(content_item, 'text'):
                        text_parts.append(content_item.text)
                elif hasattr(content_item, '__dict__') and content_item.__dict__.get('type') == 'text':
                    text_parts.append(content_item.__dict__.get('text', ''))
            
            if text_parts:
                final_result = '\n'.join(text_parts)
                logger.info(f"Tool call successful: {final_result[:100]}...")
                return final_result
            else:
                return str(result)
        else:
            return str(result)
            
    except asyncio.TimeoutError:
        error_msg = f"Timeout calling tool '{tool_name}' (exceeded 30 seconds)"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error calling tool '{tool_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg
    finally:
        # Always cleanup the connection
        try:
            await exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

async def get_available_tools(manager: MCPConnectionManager) -> Dict[str, List[Any]]:
    """Get all available tools from enabled servers"""
    tools_by_server = {}
    
    for server_name, config in manager.get_all_configs().items():
        if not config.enabled:
            continue
            
        try:
            connection_info = await manager.create_temporary_connection(server_name)
            if connection_info:
                session, exit_stack, tools = connection_info
                tools_by_server[server_name] = tools
                # Cleanup connection immediately
                await exit_stack.aclose()
        except Exception as e:
            logger.error(f"Error getting tools from {server_name}: {e}")
    
    return tools_by_server

def format_tools_for_claude(tools_by_server: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Format MCP tools for Claude API"""
    tools = []
    for server_name, server_tools in tools_by_server.items():
        for tool in server_tools:
            # Extract the actual schema - handle different attribute names
            input_schema = None
            if hasattr(tool, 'inputSchema'):
                input_schema = tool.inputSchema
            elif hasattr(tool, 'input_schema'):
                input_schema = tool.input_schema
            elif hasattr(tool, '__dict__') and 'inputSchema' in tool.__dict__:
                input_schema = tool.__dict__['inputSchema']
            
            # Ensure we have a valid schema
            if not input_schema:
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            # Get tool description
            description = ""
            if hasattr(tool, 'description'):
                description = tool.description
            elif hasattr(tool, '__dict__') and 'description' in tool.__dict__:
                description = tool.__dict__.get('description', '')
            
            # Get tool name
            tool_name = ""
            if hasattr(tool, 'name'):
                tool_name = tool.name
            elif hasattr(tool, '__dict__') and 'name' in tool.__dict__:
                tool_name = tool.__dict__.get('name', '')
            
            if tool_name:  # Only add tools with valid names
                tool_dict = {
                    "name": f"{server_name}__{tool_name}",
                    "description": f"[{server_name}] {description or tool_name}",
                    "input_schema": input_schema
                }
                tools.append(tool_dict)
    return tools

# --- Core API Interaction Function ---
async def get_anthropic_response_stream_with_mcp(
    api_messages_for_call: list,
    mcp_connections_dict: Dict[str, Any]  # Still accept old format for compatibility
):
    """Enhanced response function that handles MCP tool calls"""
    logger.info("Starting enhanced response function with MCP support")
    full_response_text = ""
    st.session_state['_current_stream_usage_data'] = None
    st.session_state['_current_stream_full_text'] = ""
    st.session_state['streaming_in_progress'] = True
    st.session_state['_stream_stopped_by_user_flag'] = False

    # Get connection manager
    manager = st.session_state.get('mcp_connection_manager')
    if not manager:
        manager = MCPConnectionManager()
        st.session_state.mcp_connection_manager = manager
    
    # Get available tools
    tools_by_server = await get_available_tools(manager)
    available_tools = format_tools_for_claude(tools_by_server)
    logger.info(f"Found {len(available_tools)} tools from {len(tools_by_server)} servers")

    try:
        # First, get Claude's response with tools
        logger.info("Creating message stream with tools")
        message_response = None
        
        async with client.messages.stream(
            max_tokens=MAX_TOKENS_OUTPUT,
            messages=api_messages_for_call,
            model=MODEL_NAME,
            tools=available_tools if available_tools else None
        ) as stream:
            logger.info("Stream created, starting to receive text")
            async for text_chunk in stream.text_stream:
                if st.session_state.get("stop_streaming", False):
                    st.session_state['_stream_stopped_by_user_flag'] = True
                    break
                full_response_text += text_chunk
                yield text_chunk
            
            # Get the final message
            logger.info("Getting final message from stream")
            message_response = await stream.get_final_message()
            logger.info(f"Final message received. Stop reason: {message_response.stop_reason if message_response else 'None'}")
            
        # Handle tool use if needed
        # Keep processing tool uses until Claude stops requesting them
        messages_for_next_call = api_messages_for_call
        max_tool_iterations = 10  # Prevent infinite loops
        tool_iteration = 0
        
        while message_response and message_response.stop_reason == "tool_use" and tool_iteration < max_tool_iterations:
            tool_iteration += 1
            logger.info(f"Tool use iteration {tool_iteration}")
            
            # Process tool calls
            tool_results = []
            
            # First, yield any text content from the assistant's message before tool use
            for content in message_response.content:
                if content.type == "text" and hasattr(content, 'text') and content.text:
                    logger.info(f"Yielding assistant text before tool: {content.text[:100]}...")
                    # Don't yield if it's already been yielded
                    if content.text not in full_response_text:
                        yield content.text
                        full_response_text += content.text
            
            for content in message_response.content:
                if content.type == "tool_use":
                    # Parse server name and tool name
                    full_tool_name = content.name
                    logger.info(f"Processing tool call: {full_tool_name}")
                    
                    if "__" in full_tool_name:
                        server_name, tool_name = full_tool_name.split("__", 1)
                        logger.info(f"Server: {server_name}, Tool: {tool_name}")
                        
                        # Show tool call info
                        tool_info = f"\n\n🔧 **[{tool_iteration}] Calling tool:** `{tool_name}` on `{server_name}`...\n"
                        logger.info(tool_info.strip())
                        yield tool_info
                        
                        # Call tool with fresh connection
                        logger.info(f"Calling tool with arguments: {content.input}")
                        result = await call_mcp_tool_with_new_connection(
                            manager, 
                            server_name, 
                            tool_name, 
                            content.input
                        )
                        logger.info(f"Tool result: {result}")
                        
                        # Display tool result in UI
                        if result:
                            # Format the result for better display
                            result_display = f"\n📤 **Tool Result:**\n```\n{result}\n```\n"
                            yield result_display
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result
                        })
                    else:
                        logger.error(f"Invalid tool name format: {full_tool_name}")
                        error_msg = f"Invalid tool name format: {full_tool_name}"
                        yield f"\n❌ **Error:** {error_msg}\n"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "is_error": True,
                            "content": error_msg
                        })
            
            # Send tool results back to Claude
            if tool_results:
                logger.info(f"Sending {len(tool_results)} tool results back to Claude")
                
                # Convert message_response.content to proper format
                assistant_content = []
                for content_item in message_response.content:
                    if content_item.type == "text":
                        assistant_content.append({
                            "type": "text",
                            "text": content_item.text if hasattr(content_item, 'text') else str(content_item)
                        })
                    elif content_item.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": content_item.id,
                            "name": content_item.name,
                            "input": content_item.input
                        })
                
                # Update messages for next call
                messages_for_next_call = messages_for_next_call + [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": tool_results}
                ]
                
                logger.info(f"Messages structure: {len(messages_for_next_call)} messages")
                
                # Get Claude's next response (might be another tool use or final answer)
                logger.info("Creating next stream to check for more tool uses")
                message_response = None
                
                async with client.messages.stream(
                    max_tokens=MAX_TOKENS_OUTPUT,
                    messages=messages_for_next_call,
                    model=MODEL_NAME,
                    tools=available_tools
                ) as next_stream:
                    logger.info("Next stream created, receiving response")
                    async for text_chunk in next_stream.text_stream:
                        if st.session_state.get("stop_streaming", False):
                            st.session_state['_stream_stopped_by_user_flag'] = True
                            break
                        full_response_text += text_chunk
                        yield text_chunk
                    
                    logger.info("Getting message from stream")
                    message_response = await next_stream.get_final_message()
                    logger.info(f"Message received. Stop reason: {message_response.stop_reason if message_response else 'None'}")
                    
                    # Update usage data
                    if message_response:
                        st.session_state['_current_stream_usage_data'] = message_response.usage
            else:
                logger.info("No tool results to send back")
                break
        
        if tool_iteration >= max_tool_iterations:
            warning_msg = f"\n\n⚠️ Reached maximum tool iterations ({max_tool_iterations}). Stopping to prevent infinite loop.\n"
            logger.warning(warning_msg)
            yield warning_msg

        st.session_state['_current_stream_full_text'] = full_response_text
        logger.info(f"Response generation complete. Total text length: {len(full_response_text)}")
        
        # Critical: yield empty string to signal end of stream
        yield ""
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        st.error(f"Error during streaming: {e}")
        yield f"\n\nError: {e}"
    finally:
        st.session_state['streaming_in_progress'] = False
        if st.session_state['_stream_stopped_by_user_flag'] or not st.session_state.get("stop_streaming", False):
            st.session_state['stop_streaming'] = False
        logger.info("Stream processing finished - exiting generator")

# --- Session State Initialization ---
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
    "current_uploaded_files": [],
    "prompt_submitted_this_run": False,
    "uploader_key_suffix": 0,
    # MCP specific state
    "mcp_server_configs": [],
    "mcp_connections": {},
    "mcp_json_input": "",
    "mcp_pending_connections": [],
    "mcp_pending_disconnections": [],
    # Add async event loop for MCP
    "_mcp_event_loop": None,
    "_mcp_thread": None,
}

for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize connection manager
if 'mcp_connection_manager' not in st.session_state:
    st.session_state.mcp_connection_manager = MCPConnectionManager()

# Cleanup function for app shutdown
def cleanup_connections():
    """Cleanup all MCP connections on app shutdown"""
    manager = st.session_state.get('mcp_connection_manager')
    if manager and hasattr(manager, 'configs'):
        # Synchronous cleanup for atexit
        for server_name in list(manager.configs.keys()):
            manager.configs.pop(server_name, None)

# Register cleanup handler
import atexit
atexit.register(cleanup_connections)

# --- UI Rendering ---
st.title("🔧 MCP Multimodal Chatbot")
st.caption(f"Model: {MODEL_NAME} | Supports text, images, PDFs, and MCP tools")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Stop button placeholder
    stop_button_placeholder = st.empty()

    # Display chat messages
    for msg_idx, message_data in enumerate(st.session_state.messages):
        with st.chat_message(message_data["role"]):
            content = message_data["content"]
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, list):
                for content_item in content:
                    item_type = content_item.get("type")
                    if item_type == "text":
                        st.markdown(content_item["text"])
                    elif item_type == "image":
                        source = content_item.get("source", {})
                        if source.get("type") == "base64":
                            media_type = source.get("media_type", "image/png")
                            base64_data = source.get("data", "")
                            st.image(f"data:{media_type};base64,{base64_data}", caption="Image")
                    elif item_type == "document":
                        st.caption("📄 PDF document")
            
            # Display usage for assistant messages
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
                        count = usage.get(token_type, 0) if isinstance(usage, dict) else getattr(usage, token_type, 0)
                        if count > 0:
                            details.append(f"{token_type.replace('_', ' ').title()}: {count}")
                    
                    if cost > 0:
                        details.append(f"Cost: ${cost:.6f}")
                    
                    if details:
                        st.caption(", ".join(details))
                elif stopped:
                    st.caption("[INFO] Streaming stopped.")

    # Control Stop Button visibility
    if st.session_state['streaming_in_progress']:
        with stop_button_placeholder.container():
            if st.button("🛑 Stop Generating", key="main_stop_button", use_container_width=True):
                st.session_state['stop_streaming'] = True
                st.session_state['_stream_stopped_by_user_flag'] = True
                st.info("Stopping generation...")
    else:
        stop_button_placeholder.empty()

    # File uploader
    current_uploader_key = f"file_uploader_widget_{st.session_state.get('uploader_key_suffix', 0)}"
    
    def on_uploader_change():
        st.session_state.current_uploaded_files = st.session_state.get(current_uploader_key) or []

    st.file_uploader(
        "📎 Attach Images or PDFs",
        type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
        accept_multiple_files=True,
        key=current_uploader_key,
        on_change=on_uploader_change
    )

    # Display staged files
    if st.session_state['current_uploaded_files']:
        st.markdown("**Staged for next message:**")
        cols = st.columns(min(len(st.session_state['current_uploaded_files']), 4))
        for idx, staged_file in enumerate(st.session_state['current_uploaded_files']):
            with cols[idx % 4]:
                if staged_file.type.startswith("image/"):
                    st.image(staged_file, caption=f"{staged_file.name[:20]}...", width=100)
                elif staged_file.type == "application/pdf":
                    st.caption(f"📄 {staged_file.name[:20]}...")
        
        if st.button("🗑️ Clear Attachments", key="clear_files_button", use_container_width=True):
            st.session_state['current_uploaded_files'] = []
            st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about images/PDFs or use MCP tools..."):
        st.session_state['prompt_submitted_this_run'] = True

        if st.session_state['streaming_in_progress']:
            st.session_state['stop_streaming'] = True
            st.info("Stopping current response...")

        # Construct user message
        user_message_content = []

        # Add files
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
                        st.error(f"Image '{uploaded_file_obj.name}' is too large ({file_size_mb:.1f}MB).")
                        continue
                    
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
                        st.error(f"PDF '{uploaded_file_obj.name}' is too large ({file_size_mb:.1f}MB).")
                        continue
                    
                    try:
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

        # Add text
        if prompt:
            user_message_content.append({"type": "text", "text": prompt})

        if not user_message_content:
            st.warning("Please enter a message or upload a file.")
            st.session_state['prompt_submitted_this_run'] = False
            st.stop()

        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_message_content})
        
        # Display user message
        with st.chat_message("user"):
            for content_item in user_message_content:
                if content_item.get("type") == "text":
                    st.markdown(content_item["text"])
                elif content_item.get("type") == "image":
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

        # Prepare API messages
        api_messages_to_send = []
        for msg_data in st.session_state.messages:
            # Create a deep copy of the message to avoid modifying the original
            msg_copy = {
                "role": msg_data["role"],
                "content": []
            }
            
            # Copy content while removing any existing cache_control
            if isinstance(msg_data["content"], list):
                for content_item in msg_data["content"]:
                    if isinstance(content_item, dict):
                        # Create a copy without cache_control
                        item_copy = {k: v for k, v in content_item.items() if k != "cache_control"}
                        msg_copy["content"].append(item_copy)
                    else:
                        msg_copy["content"].append(content_item)
            else:
                msg_copy["content"] = msg_data["content"]
            
            api_messages_to_send.append(msg_copy)

        # Add cache control ONLY to the last content of the last message
        if api_messages_to_send:
            last_message = api_messages_to_send[-1]
            if "content" in last_message and isinstance(last_message["content"], list) and last_message["content"]:
                # Add cache_control to the last content element
                last_message["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Call API with MCP support
        st.session_state['streaming_in_progress'] = True
        st.session_state['stop_streaming'] = False
        st.session_state['_stream_stopped_by_user_flag'] = False
        st.session_state['_current_stream_full_text'] = ""
        st.session_state['_current_stream_usage_data'] = None

        with st.chat_message("assistant"):
            displayed_response_text = st.write_stream(
                get_anthropic_response_stream_with_mcp(
                    api_messages_to_send,
                    st.session_state['mcp_connections']
                )
            )
            
            logger.info(f"Stream completed. Displayed text length: {len(displayed_response_text) if displayed_response_text else 0}")
            
            # Retrieve results
            retrieved_usage_info = st.session_state['_current_stream_usage_data']
            retrieved_was_stopped_by_user = st.session_state['_stream_stopped_by_user_flag']
            
            # Clean up
            st.session_state['_current_stream_usage_data'] = None
            st.session_state['_stream_stopped_by_user_flag'] = False
            st.session_state['_current_stream_full_text'] = ""

            final_assistant_message_content_text = displayed_response_text if displayed_response_text is not None else ""
            logger.info(f"Final assistant message text: {final_assistant_message_content_text[:100] if final_assistant_message_content_text else 'empty'}")

            if retrieved_was_stopped_by_user:
                if not final_assistant_message_content_text.strip():
                    final_assistant_message_content_text = "[INFO] Streaming was stopped by user."

            assistant_message_content = [{"type": "text", "text": final_assistant_message_content_text}]

            if final_assistant_message_content_text.strip() or retrieved_was_stopped_by_user:
                current_cost = 0.0
                
                if retrieved_usage_info:
                    model_costs_for_turn = get_model_costs(MODEL_NAME)

                    for token_type_api_name, cost_per_mil in model_costs_for_turn.items():
                        tokens_used_in_type = 0
                        if isinstance(retrieved_usage_info, dict):
                            tokens_used_in_type = retrieved_usage_info.get(token_type_api_name, 0)
                        elif hasattr(retrieved_usage_info, token_type_api_name):
                            tokens_used_in_type = getattr(retrieved_usage_info, token_type_api_name, 0)
                        
                        current_cost += (tokens_used_in_type / 1_000_000) * cost_per_mil

                        if tokens_used_in_type > 0:
                            st.session_state.session_total_usage[token_type_api_name] = \
                                st.session_state.session_total_usage.get(token_type_api_name, 0) + tokens_used_in_type
                    
                    st.session_state['session_total_cost'] += current_cost
                
                # Store assistant message
                usage_to_store = None
                if retrieved_usage_info:
                    if hasattr(retrieved_usage_info, 'model_dump'):
                        usage_to_store = retrieved_usage_info.model_dump()
                    elif isinstance(retrieved_usage_info, dict):
                        usage_to_store = retrieved_usage_info

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message_content,
                    "usage": usage_to_store,
                    "cost": current_cost,
                    "stopped_by_user": retrieved_was_stopped_by_user
                })
                
                logger.info(f"Added assistant message to history. Total messages: {len(st.session_state.messages)}")
                st.session_state['prompt_submitted_this_run'] = False
                st.rerun()
            else:
                logger.warning("No content from assistant and not stopped by user")
                st.session_state['prompt_submitted_this_run'] = False

# Sidebar
with col2:
    with st.sidebar:
        st.header("📊 Session Statistics")
        
        # Session totals
        st.subheader("Total Token Usage")
        total_usage_display = st.session_state.session_total_usage
        if not total_usage_display or all(val == 0 for val in total_usage_display.values()):
            st.info("No API calls yet.")
        else:
            ordered_token_keys = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
            for token_type in ordered_token_keys:
                count = total_usage_display.get(token_type, 0)
                if count > 0:
                    st.metric(
                        label=f"{token_type.replace('_', ' ').title()}",
                        value=f"{count:,}"
                    )

        st.subheader("Total Cost")
        st.metric(label="Session Cost", value=f"${st.session_state.session_total_cost:.6f}")

        st.markdown("---")

        # MCP Configuration
        st.header("🔧 MCP Configuration")
        
        # JSON input for MCP servers
        st.subheader("Add MCP Servers")
        st.caption("Enter JSON configuration for MCP servers")
        
        # Example configuration
        example_config = """[
    {
        "name": "time",
        "command": "uvx",
        "args": [
            "--from",
            "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/time",
            "mcp-server-time",
            "--local-timezone",
            "Asia/Seoul"
        ],
        "enabled": true
    },
    {
        "name": "weather",
        "command": "uvx",
        "args": [
            "--from",
            "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/weather",
            "mcp-server-weather",
            "--api-key",
            "YOUR_OPENWEATHER_API_KEY"
        ],
        "enabled": false
    }
]"""

        mcp_json_input = st.text_area(
            "MCP Server Configuration (JSON)",
            value=st.session_state.get('mcp_json_input', ''),
            height=150,
            placeholder=example_config
        )
        
        col_add, col_example = st.columns(2)
        with col_add:
            if st.button("➕ Add Servers", use_container_width=True):
                try:
                    configs = json.loads(mcp_json_input)
                    if isinstance(configs, dict):
                        configs = [configs]
                    
                    for config_data in configs:
                        config = MCPServerConfig.from_dict(config_data)
                        # Check if already exists
                        existing_names = [c.name for c in st.session_state['mcp_server_configs']]
                        if config.name not in existing_names:
                            st.session_state['mcp_server_configs'].append(config)
                            st.success(f"Added: {config.name}")
                        else:
                            st.warning(f"Server '{config.name}' already exists")
                    
                    st.session_state['mcp_json_input'] = ""
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col_example:
            if st.button("📝 Show Example", use_container_width=True):
                st.session_state['mcp_json_input'] = example_config
                st.rerun()

        st.markdown("---")

        # Connected MCP Servers
        st.subheader("MCP Servers")
        
        # Display server list
        if st.session_state['mcp_server_configs']:
            for idx, config in enumerate(st.session_state['mcp_server_configs']):
                with st.expander(f"🖥️ {config.name}", expanded=True):
                    col_toggle, col_remove = st.columns([3, 1])
                    
                    with col_toggle:
                        enabled = st.toggle(
                            "Enabled",
                            value=config.enabled,
                            key=f"mcp_toggle_{idx}"
                        )
                        if enabled != config.enabled:
                            config.enabled = enabled
                            # Mark for connection update
                            if enabled:
                                st.session_state['mcp_pending_connections'].append(config)
                            else:
                                st.session_state['mcp_pending_disconnections'].append(config.name)
                    
                    with col_remove:
                        if st.button("🗑️", key=f"remove_{idx}", help="Remove server"):
                            # Mark for disconnection if connected
                            if config.name in st.session_state['mcp_connections']:
                                st.session_state['mcp_pending_disconnections'].append(config.name)
                            st.session_state['mcp_server_configs'].pop(idx)
                            st.rerun()
                    
                    # Show connection status
                    manager = st.session_state.get('mcp_connection_manager')
                    if manager and config.name in manager.configs:
                        if config.enabled:
                            st.success("✅ Enabled (connections created on demand)")
                        else:
                            st.warning("⏸️ Disabled")
                    else:
                        st.info("⭕ Not configured")
                    
                    # Show configuration
                    st.caption(f"Command: `{config.command}`")
                    if config.args:
                        st.caption(f"Args: {' '.join(config.args)}")
        else:
            st.info("No MCP servers configured. Add one above!")

        # Apply connections button
        if st.button("🔄 Apply Changes", use_container_width=True):
            # Get or create connection manager
            manager = st.session_state.get('mcp_connection_manager')
            if not manager:
                manager = MCPConnectionManager()
                st.session_state.mcp_connection_manager = manager
            
            # Process disconnections
            for server_name in st.session_state['mcp_pending_disconnections']:
                manager.remove_config(server_name)
                # Remove from the display list
                if server_name in st.session_state['mcp_connections']:
                    del st.session_state['mcp_connections'][server_name]
            
            # Process new connections and auto-connect enabled servers
            servers_to_add = []
            
            # Add pending connections
            for config in st.session_state['mcp_pending_connections']:
                servers_to_add.append(config)
            
            # Add auto-connect servers
            for config in st.session_state['mcp_server_configs']:
                if config.enabled and config.name not in [s.name for s in servers_to_add]:
                    servers_to_add.append(config)
            
            # Add servers to manager
            for config in servers_to_add:
                manager.add_config(config.name, config)
                # Update display state
                st.session_state['mcp_connections'][config.name] = {
                    'config': config,
                    'status': 'managed'
                }
            
            # Clear pending lists
            st.session_state['mcp_pending_connections'] = []
            st.session_state['mcp_pending_disconnections'] = []
            
            st.rerun()

# Reset prompt submission flag
if not st.session_state.get("prompt_submitted_this_run", False) and not prompt:
    st.session_state['prompt_submitted_this_run'] = False 