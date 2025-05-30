import os
import base64
import mimetypes
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

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

@dataclass
class MCPConnection:
    """Active MCP connection"""
    config: MCPServerConfig
    session: ClientSession
    exit_stack: AsyncExitStack
    tools: List[Any] = field(default_factory=list)

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

# --- MCP Functions ---
async def connect_mcp_server(config: MCPServerConfig) -> Optional[MCPConnection]:
    """Connect to an MCP server"""
    try:
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env
        )
        
        exit_stack = AsyncExitStack()
        
        try:
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio_read, stdio_write = stdio_transport
        except FileNotFoundError:
            st.error(f"Command '{config.command}' not found. Please ensure it's installed and in PATH.")
            await exit_stack.aclose()
            return None
        except Exception as e:
            st.error(f"Failed to start MCP server '{config.name}': {str(e)}")
            await exit_stack.aclose()
            return None
        
        try:
            session = await exit_stack.enter_async_context(
                ClientSession(stdio_read, stdio_write)
            )
            
            await session.initialize()
            
            # Get available tools
            tools_response = await session.list_tools()
            tools = tools_response.tools if hasattr(tools_response, 'tools') else []
            
            return MCPConnection(
                config=config,
                session=session,
                exit_stack=exit_stack,
                tools=tools
            )
        except Exception as e:
            st.error(f"Failed to initialize session with MCP server '{config.name}': {str(e)}")
            await exit_stack.aclose()
            return None
            
    except Exception as e:
        st.error(f"Unexpected error connecting to MCP server '{config.name}': {str(e)}")
        return None

async def disconnect_mcp_server(connection: MCPConnection):
    """Disconnect from an MCP server"""
    try:
        await connection.exit_stack.aclose()
    except Exception as e:
        st.error(f"Error disconnecting from MCP server '{connection.config.name}': {e}")

async def call_mcp_tool(connection: MCPConnection, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool on an MCP server"""
    try:
        result = await connection.session.call_tool(tool_name, arguments)
        
        # Extract content from result
        if hasattr(result, 'content'):
            content_list = result.content
            # Concatenate all text content
            text_parts = []
            for content_item in content_list:
                if hasattr(content_item, 'type') and content_item.type == 'text':
                    if hasattr(content_item, 'text'):
                        text_parts.append(content_item.text)
                elif hasattr(content_item, '__dict__') and content_item.__dict__.get('type') == 'text':
                    text_parts.append(content_item.__dict__.get('text', ''))
            
            if text_parts:
                return '\n'.join(text_parts)
            else:
                return str(result)
        else:
            return str(result)
    except Exception as e:
        error_msg = f"Error calling tool '{tool_name}': {str(e)}"
        st.error(error_msg)
        return error_msg

def format_mcp_tools_for_claude(connections: Dict[str, MCPConnection]) -> List[Dict[str, Any]]:
    """Format MCP tools for Claude API"""
    tools = []
    for server_name, connection in connections.items():
        for tool in connection.tools:
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
    mcp_connections: Dict[str, MCPConnection]
):
    """Enhanced response function that handles MCP tool calls"""
    full_response_text = ""
    st.session_state['_current_stream_usage_data'] = None
    st.session_state['_current_stream_full_text'] = ""
    st.session_state['streaming_in_progress'] = True
    st.session_state['_stream_stopped_by_user_flag'] = False

    # Format MCP tools for Claude
    available_tools = format_mcp_tools_for_claude(mcp_connections)

    try:
        # First, get Claude's response with tools
        message_response = None
        async with client.messages.stream(
            max_tokens=MAX_TOKENS_OUTPUT,
            messages=api_messages_for_call,
            model=MODEL_NAME,
            tools=available_tools if available_tools else None
        ) as stream:
            async for text_chunk in stream.text_stream:
                if st.session_state.get("stop_streaming", False):
                    st.session_state['_stream_stopped_by_user_flag'] = True
                    break
                full_response_text += text_chunk
                yield text_chunk
            
            # Get the final message
            message_response = await stream.get_final_message()
            
        # Handle tool use if needed
        if message_response and message_response.stop_reason == "tool_use":
            # Process tool calls
            tool_results = []
            
            for content in message_response.content:
                if content.type == "tool_use":
                    # Parse server name and tool name
                    full_tool_name = content.name
                    if "__" in full_tool_name:
                        server_name, tool_name = full_tool_name.split("__", 1)
                        
                        if server_name in mcp_connections:
                            # Show tool call info
                            yield f"\n\n🔧 Calling tool: {tool_name} on {server_name}...\n"
                            
                            result = await call_mcp_tool(
                                mcp_connections[server_name],
                                tool_name,
                                content.input
                            )
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result
                            })
                        else:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "is_error": True,
                                "content": f"Server '{server_name}' not connected"
                            })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "is_error": True,
                            "content": f"Invalid tool name format: {full_tool_name}"
                        })
            
            # Send tool results back to Claude
            if tool_results:
                # Create a new messages list with tool results
                messages_with_tools = api_messages_for_call + [
                    {"role": "assistant", "content": message_response.content},
                    {"role": "user", "content": tool_results}
                ]
                
                # Get Claude's final response
                async with client.messages.stream(
                    max_tokens=MAX_TOKENS_OUTPUT,
                    messages=messages_with_tools,
                    model=MODEL_NAME,
                    tools=available_tools
                ) as final_stream:
                    async for text_chunk in final_stream.text_stream:
                        if st.session_state.get("stop_streaming", False):
                            break
                        full_response_text += text_chunk
                        yield text_chunk
                    
                    final_message = await final_stream.get_final_message()
                    st.session_state['_current_stream_usage_data'] = final_message.usage
        else:
            if message_response:
                st.session_state['_current_stream_usage_data'] = message_response.usage

        st.session_state['_current_stream_full_text'] = full_response_text
        yield ""
    except Exception as e:
        st.error(f"Error during streaming: {e}")
        yield f"\n\nError: {e}"
    finally:
        st.session_state['streaming_in_progress'] = False
        if st.session_state['_stream_stopped_by_user_flag'] or not st.session_state.get("stop_streaming", False):
            st.session_state['stop_streaming'] = False

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
}

for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

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
            api_messages_to_send.append({
                "role": msg_data["role"],
                "content": msg_data["content"]
            })

        # Add cache control
        if api_messages_to_send:
            last_message = api_messages_to_send[-1]
            if "content" in last_message and isinstance(last_message["content"], list) and last_message["content"]:
                last_content_element = last_message["content"][-1]
                if isinstance(last_content_element, dict):
                    last_content_copy = last_content_element.copy()
                    last_content_copy["cache_control"] = {"type": "ephemeral"}
                    last_message["content"][-1] = last_content_copy

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
            
            # Retrieve results
            retrieved_usage_info = st.session_state['_current_stream_usage_data']
            retrieved_was_stopped_by_user = st.session_state['_stream_stopped_by_user_flag']
            
            # Clean up
            st.session_state['_current_stream_usage_data'] = None
            st.session_state['_stream_stopped_by_user_flag'] = False
            st.session_state['_current_stream_full_text'] = ""

            final_assistant_message_content_text = displayed_response_text if displayed_response_text is not None else ""

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
                
                st.session_state['prompt_submitted_this_run'] = False
                st.rerun()
            else:
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
                    if config.name in st.session_state['mcp_connections']:
                        st.success("✅ Connected")
                        connection = st.session_state['mcp_connections'][config.name]
                        if connection.tools:
                            st.caption(f"Available tools: {len(connection.tools)}")
                            for tool in connection.tools[:3]:  # Show first 3 tools
                                st.caption(f"• {tool.name}")
                            if len(connection.tools) > 3:
                                st.caption(f"... and {len(connection.tools) - 3} more")
                    else:
                        st.info("⭕ Not connected")
                    
                    # Show configuration
                    st.caption(f"Command: `{config.command}`")
                    if config.args:
                        st.caption(f"Args: {' '.join(config.args)}")
        else:
            st.info("No MCP servers configured. Add one above!")

        # Apply connections button
        if st.button("🔄 Apply Changes", use_container_width=True):
            # Process pending connections/disconnections
            async def apply_changes():
                # Disconnections
                for server_name in st.session_state['mcp_pending_disconnections']:
                    if server_name in st.session_state['mcp_connections']:
                        with st.spinner(f"Disconnecting {server_name}..."):
                            await disconnect_mcp_server(st.session_state['mcp_connections'][server_name])
                            del st.session_state['mcp_connections'][server_name]
                
                # Connections
                for config in st.session_state['mcp_pending_connections']:
                    if config.name not in st.session_state['mcp_connections']:
                        with st.spinner(f"Connecting to {config.name}..."):
                            connection = await connect_mcp_server(config)
                            if connection:
                                st.session_state['mcp_connections'][config.name] = connection
                
                # Auto connect enabled servers
                for config in st.session_state['mcp_server_configs']:
                    if config.enabled and config.name not in st.session_state['mcp_connections']:
                        with st.spinner(f"Connecting to {config.name}..."):
                            connection = await connect_mcp_server(config)
                            if connection:
                                st.session_state['mcp_connections'][config.name] = connection
                
                # Clear pending lists
                st.session_state['mcp_pending_connections'] = []
                st.session_state['mcp_pending_disconnections'] = []
            
            # Run async function
            asyncio.run(apply_changes())
            st.rerun()

# Reset prompt submission flag
if not st.session_state.get("prompt_submitted_this_run", False) and not prompt:
    st.session_state['prompt_submitted_this_run'] = False 