import asyncio
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text

load_dotenv()

client = AsyncAnthropic()
console = Console()

async def get_anthropic_response_stream(message_history: list):
    """Sends message history to Anthropic and yields the response token by token."""
    try:
        async with client.messages.stream(
            max_tokens=1024,
            messages=message_history,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        ) as stream:
            async for text_chunk in stream.text_stream:
                yield text_chunk
    except Exception as e:
        # Ensure a newline if error occurs, so error panel is not on the same line
        console.print()
        console.print(Panel(f"[bold red]Error streaming response:[/bold red] {str(e)}", title="API Stream Error", border_style="red"))
        raise # Re-raise the exception to be caught by the caller

async def main() -> None:
    """Main function to run the chatbot."""
    console.print(Panel("[bold cyan]Welcome to the Rich Anthropic Chatbot![/bold cyan]", expand=False))
    console.print("Type 'quit' or 'exit' to end the chat.")

    message_history = [] # Initialize message history

    while True:
        user_input = Prompt.ask("[bold yellow]You[/bold yellow]")

        if user_input.lower() in ["quit", "exit"]:
            console.print(Panel("[bold cyan]Goodbye![/bold cyan]", expand=False))
            break

        if not user_input.strip():
            continue

        message_history.append({"role": "user", "content": user_input})

        console.print(Text("Assistant: ", style="bold blue")) # Print label once, outside Live

        current_markdown_content = "" # Accumulates raw markdown text for streaming display
        accumulated_response_for_history = "" # Accumulates raw text for history
        assistant_response_successful = False

        # Initial Markdown object for Live (will be empty at first)
        # Live will manage this Markdown object directly.
        markdown_display = Markdown(current_markdown_content)

        try:
            with Live(markdown_display, console=console, refresh_per_second=15, vertical_overflow="visible") as live:
                async for text_chunk in get_anthropic_response_stream(message_history):
                    current_markdown_content += text_chunk
                    accumulated_response_for_history += text_chunk
                    
                    # Update the Live display with a new Markdown object containing the updated content
                    live.update(Markdown(current_markdown_content))
                
                # If the stream completed but yielded no content, the live area would have shown Markdown("")
                # which is visually empty. This is fine.
                assistant_response_successful = True # Mark as successful if loop completes

        except Exception:
            # Error already printed by get_anthropic_response_stream
            # The console.print() below will handle moving to a new line.
            pass # The Live context will exit cleanly
        
        # Ensure a newline after Live display (or label if no Live content) finishes or on error.
        console.print()

        if assistant_response_successful and accumulated_response_for_history.strip():
            message_history.append({"role": "assistant", "content": accumulated_response_for_history})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print(Panel("\n[bold orange_red1]Chat interrupted by user. Goodbye![/bold orange_red1]", expand=False)) 