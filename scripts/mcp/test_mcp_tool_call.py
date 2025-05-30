#!/usr/bin/env python3
"""Test MCP tool calls with logging"""

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MCP_Test')

async def test_tool_call():
    """Test MCP tool call with detailed logging"""
    exit_stack = AsyncExitStack()
    
    try:
        logger.info("Creating server parameters")
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
        
        logger.info("Connecting to server")
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio_read, stdio_write = stdio_transport
        
        logger.info("Creating session")
        session = await exit_stack.enter_async_context(
            ClientSession(stdio_read, stdio_write)
        )
        
        logger.info("Initializing session")
        await session.initialize()
        
        logger.info("Listing tools")
        tools_response = await session.list_tools()
        tools = tools_response.tools if hasattr(tools_response, 'tools') else []
        
        logger.info(f"Found {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Test tool call
        logger.info("\nTesting tool call: get_current_time")
        logger.info("Arguments: {'timezone': 'Asia/Seoul'}")
        
        result = await session.call_tool("get_current_time", {"timezone": "Asia/Seoul"})
        
        logger.info(f"Raw result type: {type(result)}")
        logger.info(f"Raw result: {result}")
        
        # Extract content
        if hasattr(result, 'content'):
            logger.info(f"Result has content attribute")
            content_list = result.content
            logger.info(f"Content list type: {type(content_list)}")
            logger.info(f"Content list length: {len(content_list)}")
            
            for i, content_item in enumerate(content_list):
                logger.info(f"Content item {i}: type={type(content_item)}")
                if hasattr(content_item, 'type'):
                    logger.info(f"  - content type: {content_item.type}")
                if hasattr(content_item, 'text'):
                    logger.info(f"  - text: {content_item.text}")
        
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up")
        await exit_stack.aclose()

if __name__ == "__main__":
    logger.info("Starting MCP tool call test")
    asyncio.run(test_tool_call())
    logger.info("Test finished") 