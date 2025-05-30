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
# DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MODEL_NAME = DEFAULT_MODEL
# MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
MAX_TOKENS_OUTPUT = 4096

# Model costs
COSTS_PER_MILLION_TOKENS = {
    "claude-sonnet-4-20250514": {
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
    """Manages persistent MCP server connections"""
    
    def __init__(self):
        self.configs = {}
        self.connections = {}  # Store active connections: {server_name: (session, exit_stack, tools)}
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
    
    async def connect_server(self, server_name: str) -> bool:
        """Create persistent connection for a server"""
        config = self.get_config(server_name)
        if not config or not config.enabled:
            logger.warning(f"Cannot connect {server_name}: config not found or disabled")
            return False
        
        # Check if already connected
        if server_name in self.connections:
            logger.info(f"Server {server_name} already connected, checking health...")
            # Test connection health
            try:
                session, exit_stack, tools = self.connections[server_name]
                # Try to list tools to verify connection is still active
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                logger.info(f"Connection to {server_name} is healthy")
                return True
            except Exception as e:
                logger.warning(f"Connection to {server_name} appears unhealthy: {e}")
                # Remove unhealthy connection and reconnect
                await self.disconnect_server(server_name)
        
        logger.info(f"Connecting to MCP server: {server_name}")
        exit_stack = AsyncExitStack()
        
        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )
            
            # Start MCP server with timeout
            logger.info(f"Starting server process: {config.command} {' '.join(config.args)}")
            stdio_transport = await asyncio.wait_for(
                exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=30.0
            )
            stdio_read, stdio_write = stdio_transport
            
            # Create session with timeout
            logger.info(f"Creating session for {server_name}")
            session = await asyncio.wait_for(
                exit_stack.enter_async_context(ClientSession(stdio_read, stdio_write)),
                timeout=10.0
            )
            
            # Initialize session with timeout
            logger.info(f"Initializing session for {server_name}")
            await asyncio.wait_for(session.initialize(), timeout=10.0)
            
            # Get tools with timeout
            logger.info(f"Getting tools for {server_name}")
            tools_response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
            tools = tools_response.tools if hasattr(tools_response, 'tools') else []
            
            # Store connection
            with self.lock:
                self.connections[server_name] = (session, exit_stack, tools)
            
            logger.info(f"Successfully connected to {server_name} with {len(tools)} tools")
            
            # Log available tools for debugging
            tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
            logger.info(f"Available tools for {server_name}: {tool_names}")
            
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to {server_name}")
            await exit_stack.aclose()
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            await exit_stack.aclose()
            return False
    
    async def check_connection_health(self, server_name: str) -> bool:
        """Check if a connection is still healthy"""
        connection_info = self.get_connection(server_name)
        if not connection_info:
            return False
        
        session, exit_stack, tools = connection_info
        try:
            # Try to list tools to verify connection
            await asyncio.wait_for(session.list_tools(), timeout=5.0)
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {server_name}: {e}")
            return False
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from a server"""
        with self.lock:
            if server_name in self.connections:
                session, exit_stack, tools = self.connections.pop(server_name)
                logger.info(f"Disconnecting from {server_name}")
                try:
                    await exit_stack.aclose()
                    logger.info(f"Successfully disconnected from {server_name}")
                except Exception as e:
                    logger.error(f"Error disconnecting from {server_name}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all servers"""
        server_names = list(self.connections.keys())
        for server_name in server_names:
            await self.disconnect_server(server_name)
    
    def get_connection(self, server_name: str) -> Optional[Tuple[Any, Any, List[Any]]]:
        """Get existing connection for a server"""
        with self.lock:
            return self.connections.get(server_name)
    
    def get_all_tools(self) -> Dict[str, List[Any]]:
        """Get all tools from connected servers"""
        tools_by_server = {}
        with self.lock:
            for server_name, (session, exit_stack, tools) in self.connections.items():
                config = self.configs.get(server_name)
                if config and config.enabled:
                    tools_by_server[server_name] = tools
        return tools_by_server
    
    def is_connected(self, server_name: str) -> bool:
        """Check if server is connected"""
        with self.lock:
            return server_name in self.connections
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names"""
        with self.lock:
            return list(self.connections.keys())

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

# --- Helper Functions for Connection Management ---
async def connect_mcp_servers(manager: MCPConnectionManager, configs: List[MCPServerConfig]) -> Dict[str, bool]:
    """Connect to multiple MCP servers"""
    results = {}
    for config in configs:
        if config.enabled:
            success = await manager.connect_server(config.name)
            results[config.name] = success
        else:
            results[config.name] = False
    return results

async def disconnect_mcp_servers(manager: MCPConnectionManager, server_names: List[str]):
    """Disconnect from multiple MCP servers"""
    for server_name in server_names:
        await manager.disconnect_server(server_name)

async def check_all_connections_health(manager: MCPConnectionManager) -> Dict[str, bool]:
    """Check health of all connections"""
    health_status = {}
    for server_name in manager.get_connected_servers():
        health_status[server_name] = await manager.check_connection_health(server_name)
    return health_status

# --- MCP Functions ---
async def call_mcp_tool_with_persistent_connection(manager: MCPConnectionManager, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool on an MCP server using persistent connection"""
    logger.info(f"Calling tool {tool_name} on server {server_name}")
    
    # Get existing connection
    connection_info = manager.get_connection(server_name)
    if not connection_info:
        error_msg = f"No active connection to server '{server_name}'. Please ensure the server is connected."
        logger.error(error_msg)
        return error_msg
    
    session, exit_stack, tools = connection_info
    
    # Validate that the tool exists
    available_tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
    if tool_name not in available_tool_names:
        error_msg = f"Tool '{tool_name}' not found in server '{server_name}'. Available tools: {available_tool_names}"
        logger.error(error_msg)
        return error_msg
    
    try:
        # Check if session is still active by testing a simple operation
        logger.info(f"Validating connection to {server_name}")
        
        # Call the tool with increased timeout
        logger.info(f"Executing tool call: {tool_name} with args: {arguments}")
        
        # Increase timeout to 60 seconds for potentially slow operations
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=60.0
        )
        
        # Extract content from result
        if hasattr(result, 'content') and result.content:
            content_list = result.content
            text_parts = []
            for content_item in content_list:
                if hasattr(content_item, 'type') and content_item.type == 'text':
                    if hasattr(content_item, 'text'):
                        text_parts.append(content_item.text)
                elif hasattr(content_item, '__dict__'):
                    item_dict = content_item.__dict__
                    if item_dict.get('type') == 'text':
                        text_parts.append(item_dict.get('text', ''))
            
            if text_parts:
                final_result = '\n'.join(text_parts)
                logger.info(f"Tool call successful: {final_result[:100]}...")
                return final_result
            else:
                # If no text content, return the raw result
                result_str = str(result)
                logger.info(f"Tool call returned non-text result: {result_str[:100]}...")
                return result_str
        else:
            # No content attribute or empty content
            result_str = str(result)
            logger.info(f"Tool call returned result without content: {result_str[:100]}...")
            return result_str
            
    except asyncio.TimeoutError:
        error_msg = f"Timeout calling tool '{tool_name}' on server '{server_name}' (exceeded 60 seconds)"
        logger.error(error_msg)
        
        # Try to reconnect the server for next time
        logger.info(f"Attempting to reconnect {server_name} due to timeout...")
        try:
            await manager.disconnect_server(server_name)
            config = manager.get_config(server_name)
            if config:
                await manager.connect_server(server_name)
                logger.info(f"Successfully reconnected to {server_name}")
        except Exception as reconnect_error:
            logger.error(f"Failed to reconnect to {server_name}: {reconnect_error}")
        
        return error_msg
    except Exception as e:
        error_msg = f"Error calling tool '{tool_name}' on server '{server_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Check if it's a connection issue and try to reconnect
        if "connection" in str(e).lower() or "closed" in str(e).lower():
            logger.info(f"Detected connection issue with {server_name}, attempting to reconnect...")
            try:
                await manager.disconnect_server(server_name)
                config = manager.get_config(server_name)
                if config:
                    reconnect_success = await manager.connect_server(server_name)
                    if reconnect_success:
                        logger.info(f"Successfully reconnected to {server_name}")
                        # Get the new session after reconnection
                        new_connection_info = manager.get_connection(server_name)
                        if new_connection_info:
                            new_session, _, _ = new_connection_info
                            # Retry the tool call once with new session
                            logger.info(f"Retrying tool call: {tool_name}")
                            retry_result = await asyncio.wait_for(
                                new_session.call_tool(tool_name, arguments),
                                timeout=60.0
                            )
                            
                            # Process retry result same way as original
                            if hasattr(retry_result, 'content') and retry_result.content:
                                content_list = retry_result.content
                                text_parts = []
                                for content_item in content_list:
                                    if hasattr(content_item, 'type') and content_item.type == 'text':
                                        if hasattr(content_item, 'text'):
                                            text_parts.append(content_item.text)
                                
                                if text_parts:
                                    final_result = '\n'.join(text_parts)
                                    logger.info(f"Retry successful: {final_result[:100]}...")
                                    return final_result
                            
                            return str(retry_result)
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect and retry: {reconnect_error}")
        
        return error_msg

async def get_available_tools(manager: MCPConnectionManager) -> Dict[str, List[Any]]:
    """Get all available tools from connected servers"""
    return manager.get_all_tools()

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
                    logger.info(f"Found assistant text before tool: {content.text[:100]}...")
                    # Always yield the text if it's not empty, regardless of whether it's in full_response_text
                    if content.text.strip():
                        yield "\n\n" + content.text + "\n"
                        full_response_text += "\n\n" + content.text + "\n"
            
            for content in message_response.content:
                if content.type == "tool_use":
                    # Parse server name and tool name
                    full_tool_name = content.name
                    logger.info(f"Processing tool call: {full_tool_name}")
                    
                    if "__" in full_tool_name:
                        server_name, tool_name = full_tool_name.split("__", 1)
                        logger.info(f"Server: {server_name}, Tool: {tool_name}")
                        
                        # Show tool call info with arguments
                        tool_info = f"\n\n🔧 **[{tool_iteration}] Calling tool:** `{tool_name}` on `{server_name}`\n"
                        
                        # Format and display tool arguments
                        try:
                            import json
                            args_formatted = json.dumps(content.input, indent=2, ensure_ascii=False)
                            tool_info += f"\n📥 **Input arguments:**\n```json\n{args_formatted}\n```\n"
                        except:
                            # Fallback if JSON formatting fails
                            tool_info += f"\n📥 **Input arguments:** `{content.input}`\n"
                        
                        logger.info(tool_info.strip())
                        yield tool_info
                        
                        # Call tool with persistent connection
                        logger.info(f"Calling tool with arguments: {content.input}")
                        result = await call_mcp_tool_with_persistent_connection(
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
                
                # Add a separator for clarity
                yield "\n---\n"
                
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
    # Chat reset confirmation
    "show_reset_confirm": False,
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
    if manager:
        # Run cleanup in a new event loop since we might not have one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.disconnect_all())
            loop.close()
            logger.info("All MCP connections cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Register cleanup handler
import atexit
atexit.register(cleanup_connections)

# --- UI Rendering ---
# Get connected servers count for title
manager = st.session_state.get('mcp_connection_manager')
connected_count = len(manager.get_connected_servers()) if manager else 0
if connected_count > 0:
    st.title(f"🔧 MCP Multimodal Chatbot ({connected_count} servers)")
else:
    st.title("🔧 MCP Multimodal Chatbot")
st.caption(f"Model: {MODEL_NAME} | Supports text, images, PDFs, and MCP tools")

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

    # Chat Reset Button
    st.markdown("---")
    st.subheader("🔄 Chat Controls")
    
    col_reset, col_confirm = st.columns([1, 1])
    with col_reset:
        if st.button("🗑️ Reset Chat", use_container_width=True, help="Clear chat history and statistics (keeps MCP servers)"):
            st.session_state['show_reset_confirm'] = True
            st.rerun()
    
    with col_confirm:
        if st.session_state.get('show_reset_confirm', False):
            if st.button("✅ Confirm", use_container_width=True, type="primary"):
                # Reset chat-related session state while preserving MCP configurations
                chat_reset_keys = [
                    'messages',
                    'session_total_usage', 
                    'session_total_cost',
                    'session_total_hypothetical_cost',
                    'current_uploaded_files',
                    'streaming_in_progress',
                    'stop_streaming',
                    '_stream_stopped_by_user_flag',
                    '_current_stream_usage_data',
                    '_current_stream_full_text',
                    'prompt_submitted_this_run',
                    'uploader_key_suffix'
                ]
                
                for key in chat_reset_keys:
                    if key in default_session_state_values:
                        st.session_state[key] = default_session_state_values[key]
                
                # Reset uploader key to clear file uploader
                st.session_state.uploader_key_suffix = st.session_state.get('uploader_key_suffix', 0) + 1
                
                # Hide confirmation and show success
                st.session_state['show_reset_confirm'] = False
                
                # Show what was preserved
                manager = st.session_state.get('mcp_connection_manager')
                if manager:
                    connected_count = len(manager.get_connected_servers())
                    if connected_count > 0:
                        st.success(f"Chat cleared! {connected_count} MCP servers remain connected.")
                    else:
                        st.success("Chat cleared! (No MCP servers were connected)")
                else:
                    st.success("Chat cleared!")
                st.rerun()
    
    # Cancel confirmation if it's showing
    if st.session_state.get('show_reset_confirm', False):
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state['show_reset_confirm'] = False
            st.rerun()

    st.markdown("---")

    # MCP Configuration
    st.header("🔧 MCP Configuration")
    
    # Connection Status Summary
    manager = st.session_state.get('mcp_connection_manager')
    if manager:
        connected_servers = manager.get_connected_servers()
        if connected_servers:
            col_status, col_health = st.columns([3, 1])
            with col_status:
                st.subheader("📡 Connected Servers")
            with col_health:
                if st.button("🔍 Check Health", help="Check if all connections are healthy"):
                    with st.spinner("Checking connections..."):
                        health_results = run_async(check_all_connections_health(manager))
                        for server_name, is_healthy in health_results.items():
                            if is_healthy:
                                st.success(f"✅ {server_name}: Healthy")
                            else:
                                st.error(f"❌ {server_name}: Unhealthy")
                                # Try to reconnect unhealthy servers
                                config = manager.get_config(server_name)
                                if config:
                                    with st.spinner(f"Reconnecting {server_name}..."):
                                        success = run_async(manager.connect_server(server_name))
                                        if success:
                                            st.success(f"🔄 {server_name}: Reconnected")
                                        else:
                                            st.error(f"💥 {server_name}: Failed to reconnect")
            
            total_tools = 0
            for server_name in connected_servers:
                connection_info = manager.get_connection(server_name)
                if connection_info:
                    _, _, tools = connection_info
                    tool_count = len(tools)
                    total_tools += tool_count
                    
                    # Show tool details
                    tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
                    st.caption(f"🖥️ **{server_name}**: {tool_count} tools - {', '.join(tool_names)}")
            
            st.info(f"**Total: {len(connected_servers)} servers, {total_tools} tools available**")
        else:
            st.warning("No servers currently connected")
    
    st.markdown("---")

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
    "name": "sequential-thinking",
    "command": "npx",
    "args": [
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
    ],
    "enabled": true
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
                if manager and manager.is_connected(config.name):
                    if config.enabled:
                        st.success("✅ Connected")
                    else:
                        st.warning("⚠️ Connected but disabled")
                elif config.enabled:
                    st.error("❌ Not connected")
                else:
                    st.info("⏸️ Disabled")
                
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
        
        # Show progress
        with st.spinner("Applying changes..."):
            # Process disconnections first
            servers_to_disconnect = []
            for server_name in st.session_state['mcp_pending_disconnections']:
                if manager.is_connected(server_name):
                    servers_to_disconnect.append(server_name)
                manager.remove_config(server_name)
                # Remove from the display list
                if server_name in st.session_state['mcp_connections']:
                    del st.session_state['mcp_connections'][server_name]
            
            # Disconnect servers
            if servers_to_disconnect:
                run_async(disconnect_mcp_servers(manager, servers_to_disconnect))
                st.success(f"Disconnected from {len(servers_to_disconnect)} servers")
            
            # Process new connections and auto-connect enabled servers
            servers_to_connect = []
            
            # Add pending connections
            for config in st.session_state['mcp_pending_connections']:
                servers_to_connect.append(config)
            
            # Add auto-connect servers (enabled servers not yet connected)
            for config in st.session_state['mcp_server_configs']:
                if config.enabled and not manager.is_connected(config.name):
                    if config.name not in [s.name for s in servers_to_connect]:
                        servers_to_connect.append(config)
            
            # Add server configs to manager
            for config in servers_to_connect:
                manager.add_config(config.name, config)
            
            # Connect to servers
            if servers_to_connect:
                connection_results = run_async(connect_mcp_servers(manager, servers_to_connect))
                
                # Update display state based on connection results
                for config in servers_to_connect:
                    if connection_results.get(config.name, False):
                        st.session_state['mcp_connections'][config.name] = {
                            'config': config,
                            'status': 'connected'
                        }
                        st.success(f"✅ Connected to {config.name}")
                    else:
                        st.error(f"❌ Failed to connect to {config.name}")
        
        # Clear pending lists
        st.session_state['mcp_pending_connections'] = []
        st.session_state['mcp_pending_disconnections'] = []
        
        st.rerun()

# Reset prompt submission flag
if not st.session_state.get("prompt_submitted_this_run", False) and not prompt:
    st.session_state['prompt_submitted_this_run'] = False 