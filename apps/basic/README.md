# FastAPI Chatbot with Token Counting & Cost

This application is a web-based chatbot that uses the Anthropic API (Claude models) to provide responses. It features real-time streaming of answers, token usage tracking per message, cost estimation, and the ability to stop message generation.

## Prerequisites

1.  **Python**: Ensure you have Python 3.7+ installed.
2.  **pip**: Python package installer.
3.  **Anthropic API Key**: You need an API key from Anthropic. Create a file named `.env` in the root directory of this project (i.e., `/home/lsw91/Workspace/ai-playground-1/.env`) with the following content:
    ```env
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    # You can optionally set a default model, e.g.:
    # ANTHROPIC_MODEL=claude-3-haiku-20240307
    ```
    Replace `your_anthropic_api_key_here` with your actual key.

## Setup and Installation

1.  **Navigate to the Project Root**: Open your terminal and go to the root directory of this AI playground project:
    ```bash
    cd /home/lsw91/Workspace/ai-playground-1
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**: Install the required Python packages using the `requirements.txt` file located within the `apps/basic` directory:
    ```bash
    pip install -r apps/basic/requirements.txt
    ```

## Running the Application

1.  **Start the FastAPI Server**: From the project root directory (`/home/lsw91/Workspace/ai-playground-1`), run the following command:
    ```bash
    uvicorn apps.basic.main:app --reload --port 8001
    ```
    *   `uvicorn` is the ASGI server that runs the FastAPI application.
    *   `apps.basic.main:app` tells uvicorn where to find the FastAPI `app` instance (in `apps/basic/main.py`).
    *   `--reload` enables auto-reloading of the server when code changes are detected (useful for development).
    *   `--port 8001` specifies that the server should run on port 8001. You can change this if the port is already in use.

2.  **Access the Chatbot**: Open your web browser and navigate to:
    ```
    http://localhost:8001
    ```

You should now see the chatbot interface and be able to interact with it.

## Application Structure

*   `apps/basic/main.py`: The FastAPI backend server handling API calls and WebSocket communication.
*   `apps/basic/static/index.html`: The main HTML file for the user interface.
*   `apps/basic/static/script.js`: Client-side JavaScript for interactivity and WebSocket handling.
*   `apps/basic/static/style.css`: CSS for styling the application.
*   `apps/basic/requirements.txt`: Lists the Python dependencies.
*   `apps/basic/README.md`: This file. 