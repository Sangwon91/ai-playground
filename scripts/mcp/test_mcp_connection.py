#!/usr/bin/env python3
"""Test script to verify MCP server connections"""

import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys

async def test_mcp_server():
    """Test connection to MCP time server"""
    print("Testing MCP time server connection...")
    
    # Test configuration - using Git-based which is more reliable
    config = {
        "name": "Time Server (From Git)",
        "command": "uvx", 
        "args": [
            "--from",
            "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/time",
            "mcp-server-time",
            "--local-timezone",
            "Asia/Seoul"
        ],
    }
    
    print(f"\n{'='*50}")
    print(f"Testing: {config['name']}")
    print(f"Command: {config['command']} {' '.join(config['args'])}")
    print('='*50)
    
    exit_stack = AsyncExitStack()
    
    try:
        server_params = StdioServerParameters(
            command=config["command"],
            args=config["args"],
            env=None
        )
        
        # Connect to server
        print("Connecting to server...")
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio_read, stdio_write = stdio_transport
        
        session = await exit_stack.enter_async_context(
            ClientSession(stdio_read, stdio_write)
        )
        
        # Initialize session
        await session.initialize()
        print("✅ Session initialized successfully")
        
        # List tools
        tools_response = await session.list_tools()
        tools = tools_response.tools if hasattr(tools_response, 'tools') else []
        
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            name = tool.name if hasattr(tool, 'name') else 'Unknown'
            desc = tool.description if hasattr(tool, 'description') else 'No description'
            print(f"  - {name}: {desc}")
            
            # Show input schema
            if hasattr(tool, 'inputSchema'):
                print(f"    Input schema: {tool.inputSchema}")
        
        # Test calling tools
        if tools:
            print("\n" + "="*50)
            print("Testing tool calls...")
            print("="*50)
            
            # Test get_current_time
            print("\n1. Testing get_current_time with Asia/Seoul:")
            try:
                result = await session.call_tool("get_current_time", {"timezone": "Asia/Seoul"})
                print(f"   Result: {result}")
            except Exception as e:
                print(f"   Error: {e}")
            
            # Test get_current_time with UTC
            print("\n2. Testing get_current_time with UTC:")
            try:
                result = await session.call_tool("get_current_time", {"timezone": "UTC"})
                print(f"   Result: {result}")
            except Exception as e:
                print(f"   Error: {e}")
                
            # Test convert_time if available
            if any(tool.name == 'convert_time' for tool in tools if hasattr(tool, 'name')):
                print("\n3. Testing convert_time:")
                try:
                    result = await session.call_tool("convert_time", {
                        "source_timezone": "Asia/Seoul",
                        "time": "14:30",
                        "target_timezone": "America/New_York"
                    })
                    print(f"   Result: {result}")
                except Exception as e:
                    print(f"   Error: {e}")
        
        await exit_stack.aclose()
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Command '{config['command']}' not found")
        print("   Please install uv: pip install uv")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await exit_stack.aclose()
        except:
            pass

    print("\n\nRecommendations for Streamlit app:")
    print("1. Use the Git-based uvx command format for reliable installation")
    print("2. Always specify proper IANA timezone names (e.g., 'Asia/Seoul' not 'KST')")
    print("3. The time server provides 'get_current_time' and 'convert_time' tools")

if __name__ == "__main__":
    asyncio.run(test_mcp_server()) 