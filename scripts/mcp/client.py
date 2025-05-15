import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv()  # load environment variables from .env

console = Console()
MODEL_NAME = "claude-3-5-sonnet-latest"  # 모델명을 글로벌 변수로 정의

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, command: str):
        """Connect to an MCP server
        
        Args:
            command: Command to run the server
        """
        # is_python = server_script_path.endswith('.py')
        # is_js = server_script_path.endswith('.js')
        # if not (is_python or is_js):
        #     raise ValueError("Server script must be a .py or .js file")
            
        # command = "python" if is_python else "node"
        console.log(f"Connecting to server with command: [bold cyan]{command}[/bold cyan]")
        tokens = command.split(' ')
        command = tokens[0]
        args = tokens[1:]
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )
        
        console.log(f"Connecting to server with command: [bold cyan]{command}[/bold cyan] {' '.join(args)}")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        console.print(response)
        tools = response.tools
        tool_names = [tool.name for tool in tools]
        console.print("\n[bold green]Connected to server with tools:[/bold green]", style="bold")
        for tool in tool_names:
            console.print(f"  • [cyan]{tool}[/cyan]")

    async def process_query(self, query: str, prev_messages: list[dict] | None = None) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        if prev_messages:
            messages = prev_messages + messages

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        console.log("Sending query to Claude...")
        response = self.anthropic.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                console.log(f"Executing tool: [bold cyan]{tool_name}[/bold cyan]")
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # Log tool output
                console.log(f"Tool output: [bold green]{result.content}[/bold green]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                console.log("Getting follow-up response from Claude...")
                response = self.anthropic.messages.create(
                    model=MODEL_NAME,
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text), messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        console.print(Panel.fit(
            "[bold blue]MCP Client Started![/bold blue]\nType your queries or 'quit' to exit.",
            border_style="green"
        ))
        
        messages = []
        while True:
            try:
                query = console.input("\n[bold yellow]Query:[/bold yellow] ").strip()
                
                if query.lower() == 'quit':
                    console.print("[bold red]Exiting chat...[/bold red]")
                    break
                    
                with console.status("[bold green]Processing query...[/bold green]"):
                    response, messages = await self.process_query(
                        query, prev_messages=messages
                    )
                
                console.print(Panel(
                    Text(response, style="cyan"),
                    title="Response",
                    border_style="blue"
                ))
                    
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}", style="red")
    
    async def cleanup(self):
        """Clean up resources"""
        console.log("Cleaning up resources...")
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage: python client.py <command>[/bold red]")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())