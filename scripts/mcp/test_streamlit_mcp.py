#!/usr/bin/env python3
"""Simple test for Streamlit MCP integration"""

import streamlit as st
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import concurrent.futures
import json

st.title("MCP Integration Test")

def run_async(coro):
    """Run async function in a way compatible with Streamlit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (Streamlit case), create a task
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # If no loop is running, create one
            return asyncio.run(coro)
    except RuntimeError:
        # Fallback: create new event loop in thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

async def test_mcp():
    """Test MCP connection"""
    exit_stack = AsyncExitStack()
    
    try:
        st.write("1. Creating server parameters...")
        server_params = StdioServerParameters(
            command="uvx",
            args=[
                "--from",
                "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/time",
                "mcp-server-time",
                "--local-timezone",
                "Asia/Seoul"
            ],
            env=None
        )
        
        st.write("2. Connecting to server...")
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio_read, stdio_write = stdio_transport
        
        st.write("3. Creating session...")
        session = await exit_stack.enter_async_context(
            ClientSession(stdio_read, stdio_write)
        )
        
        st.write("4. Initializing session...")
        await session.initialize()
        st.success("✅ Session initialized!")
        
        st.write("5. Listing tools...")
        tools_response = await session.list_tools()
        tools = tools_response.tools if hasattr(tools_response, 'tools') else []
        
        st.write(f"Found {len(tools)} tools:")
        for tool in tools:
            st.write(f"- {tool.name}: {tool.description}")
        
        st.write("6. Testing tool call...")
        result = await session.call_tool("get_current_time", {"timezone": "Asia/Seoul"})
        
        # Extract text from result
        if hasattr(result, 'content'):
            content_list = result.content
            for content_item in content_list:
                if hasattr(content_item, 'text'):
                    st.success(f"Tool result: {content_item.text}")
        
        return True, None
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return False, error_msg
    finally:
        st.write("7. Cleaning up...")
        try:
            await exit_stack.aclose()
        except Exception as e:
            st.warning(f"Cleanup warning: {e}")

if st.button("Test MCP Connection"):
    with st.spinner("Testing..."):
        success, error = run_async(test_mcp())
        
        if success:
            st.success("✅ All tests passed!")
        else:
            st.error("❌ Test failed!")
            if error:
                st.code(error)

st.markdown("---")
st.markdown("### Instructions")
st.markdown("""
1. Click the 'Test MCP Connection' button
2. Watch the progress messages
3. If successful, you should see the current time
4. If it fails, check the error message
""") 