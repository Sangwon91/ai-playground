import sys
import os
import base64
import mimetypes
import asyncio
import json # For message passing with JS

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QLabel, QFileDialog, QSizePolicy, QMessageBox, QGridLayout
)
from PySide6.QtGui import QColor, QPalette, QDesktopServices, QPixmap
from PySide6.QtCore import Qt, Slot, QThread, Signal, QUrl, QDateTime
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView

from anthropic import (
    AsyncAnthropic, APIError, APIConnectionError, RateLimitError, APIStatusError
)
from dotenv import load_dotenv
import mistune # For Markdown

# Load environment variables
load_dotenv()


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Configuration & Constants ---
DEFAULT_MODEL = "claude-sonnet-4-20250514" # Updated default to Sonnet for better multimodal
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
MAX_TOKENS_OUTPUT = 4096 # Increased max tokens for potentially larger multimodal responses

# Model-specific costs per million tokens
COSTS_PER_MILLION_TOKENS = {
    "claude-3-5-sonnet-20240620": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75, # Cost for creating a cache from input
        "cache_read_input_tokens": 0.30,   # Cost for reading from cache (considered as input)
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
    "claude-3-5-haiku-20241022": { 
        "input_tokens": 0.8, # Assuming costs, adjust if known
        "output_tokens": 4,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 0.08,
    },
    # Older models kept for reference if needed
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
}

DEFAULT_MODEL_COSTS = COSTS_PER_MILLION_TOKENS.get(
    MODEL_NAME, # Use current model for default costs
    { # Fallback default costs if model not in dict (e.g. a new one)
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
)

# --- Anthropic API Worker Thread ---
class AnthropicWorker(QThread):
    response_chunk = Signal(str)
    response_error = Signal(str)
    response_finished = Signal(dict)  # For usage data and final status

    def __init__(self, api_key, model_name, messages, max_tokens):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.messages = messages
        self.max_tokens = max_tokens
        self.client = None # Initialized in run() for thread safety with asyncio
        self._is_running = True

    async def get_response_async(self):
        self.client = AsyncAnthropic(api_key=self.api_key)
        try:
            async with self.client.messages.stream(
                max_tokens=self.max_tokens,
                messages=self.messages,
                model=self.model_name,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    if not self._is_running:
                        self.response_error.emit("Streaming stopped by user.")
                        return
                    self.response_chunk.emit(text_chunk)
                
                if self._is_running:
                    final_message = await stream.get_final_message()
                    self.response_finished.emit({
                        "usage": final_message.usage.model_dump(),
                        "stop_reason": final_message.stop_reason
                    })
                else: # Stream was stopped before completion
                    self.response_finished.emit({"usage": {}, "stop_reason": "user_request"})
                    
        except APIConnectionError as e:
            self.response_error.emit(f"API Connection Error: {e.__cause__}")
        except RateLimitError as e:
            self.response_error.emit(f"API Rate Limit Exceeded: {e}")
        except APIStatusError as e:
            self.response_error.emit(f"API Status Error (code {e.status_code}): {e.response}")
        except APIError as e:
            self.response_error.emit(f"Generic API Error: {e}")
        except Exception as e:
            self.response_error.emit(f"An unexpected error occurred: {e}")
        finally:
            if self.client: # Close client session if it was opened
                await self.client.close()


    def run(self):
        try:
            asyncio.run(self.get_response_async())
        except Exception as e:
            # This might catch errors if asyncio.run itself fails or if client init fails
            self.response_error.emit(f"Error running async task: {e}")
    
    def stop(self):
        self._is_running = False

# --- Custom WebEnginePage to handle links and JS communication ---
class ChatWebEnginePage(QWebEnginePage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile().setHttpAcceptLanguage("en-US,en;q=0.9")
        self.profile().scripts().insert(self._create_mathjax_script())

    def _create_mathjax_script(self):
        # Script to load MathJax for LaTeX rendering
        script_content = """
        (function() {
            if (!window.MathJax) {
                var script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
                script.async = true;
                script.id = 'MathJax-script';
                // Ensure head exists, or wait for DOMContentLoaded if run very early
                if (document.head) {
                    document.head.appendChild(script);
                } else {
                    document.addEventListener('DOMContentLoaded', function() { document.head.appendChild(script); });
                }

                window.MathJax = {
                    tex: {
                        inlineMath: [['$', '$'], ['\\(', '\\)']],
                        displayMath: [['$$', '$$'], ['\\[', '\\]']],
                        processEscapes: true
                    },
                    startup: {
                        ready: () => {
                            MathJax.startup.defaultReady();
                            // Listen for new messages to typeset
                            document.addEventListener('newMessageForMathJax', function() {
                                MathJax.typesetPromise();
                            });
                        }
                    }
                };
            }
        })();
        """
        script = QWebEngineScript()
        script.setSourceCode(script_content)
        script.setName("MathJaxLoader")
        script.setInjectionPoint(QWebEngineScript.DocumentReady) # Changed injection point
        script.setRunsOnSubFrames(False)
        script.setWorldId(QWebEngineScript.MainWorld)
        return script

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if nav_type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False # Prevent QWebEngineView from navigating
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)

# --- Main Chat Window ---
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PySide6 Multimodal Chatbot ({MODEL_NAME})")
        self.setGeometry(100, 100, 900, 750) # Increased height slightly

        self.chat_history_api = [] # Stores messages for API calls
        self.uploaded_files_data = [] # {name, type, data (base64), path_for_display}
        self.anthropic_worker = None
        self.current_assistant_message_id = None
        self.markdown_parser = mistune.create_markdown(renderer=mistune.HTMLRenderer(escape=False)) # Allow HTML for images/etc.
        self.created_assistant_message_ids = set() # Keep track of created assistant message divs

        # Session statistics
        self.session_total_usage = {}
        self.session_total_cost = 0.0
        self.session_total_hypothetical_cost = 0.0

        self._init_ui()
        self._apply_styles()

        self.initial_page_loaded = False
        self.chat_page.loadFinished.connect(self._handle_initial_load_finished)

        if not ANTHROPIC_API_KEY:
            self._show_error_popup("ANTHROPIC_API_KEY not found. Please set it in your .env file.")
            # Buttons will remain disabled due to initial_page_loaded being false / ANTHROPIC_API_KEY check
        else:
            # Attempt to load HTML. Buttons enable on _handle_initial_load_finished
            self._init_chat_html() # Load initial HTML structure

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Chat Display Area (QWebEngineView)
        self.chat_view = QWebEngineView()
        self.chat_page = ChatWebEnginePage(self.chat_view)
        self.chat_view.setPage(self.chat_page)
        # self._init_chat_html() # Moved to __init__ after connecting loadFinished
        main_layout.addWidget(self.chat_view, 1)

        # Staged files display area
        self.staged_files_label = QLabel("Staged Files:")
        self.staged_files_label.setVisible(False)
        main_layout.addWidget(self.staged_files_label)

        self.staged_files_display_area = QWidget()
        self.staged_files_layout = QHBoxLayout(self.staged_files_display_area)
        self.staged_files_display_area.setVisible(False)
        self.staged_files_display_area.setFixedHeight(70)
        main_layout.addWidget(self.staged_files_display_area)

        # Input Area
        input_controls_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Ask about images/PDFs or send a message... (Shift+Enter for newline)")
        self.input_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_field.setFixedHeight(60) # Adjusted height
        self.input_field.keyPressEvent = self._handle_input_key_press
        input_controls_layout.addWidget(self.input_field, 3) # Give more stretch to input

        # Buttons Box (Attach, Send, Stop, Reset)
        buttons_box_layout = QVBoxLayout()
        
        action_buttons_layout = QHBoxLayout()
        self.attach_button = QPushButton("📎 Attach")
        self.attach_button.clicked.connect(self._attach_file)
        self.attach_button.setEnabled(False) # Disable initially
        action_buttons_layout.addWidget(self.attach_button)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_message_slot)
        self.send_button.setEnabled(False) # Disable initially
        action_buttons_layout.addWidget(self.send_button)
        buttons_box_layout.addLayout(action_buttons_layout)

        control_buttons_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_generation)
        self.stop_button.setEnabled(False)
        control_buttons_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("🔄 Reset Chat")
        self.reset_button.clicked.connect(self._reset_chat_slot)
        self.reset_button.setEnabled(False) # Disable initially, enable with other controls
        control_buttons_layout.addWidget(self.reset_button)
        buttons_box_layout.addLayout(control_buttons_layout)
        
        input_controls_layout.addLayout(buttons_box_layout, 1) # Less stretch for buttons area
        main_layout.addLayout(input_controls_layout)

        # Usage and Cost Display Area
        self.usage_cost_area = QWidget()
        usage_cost_layout = QGridLayout(self.usage_cost_area)
        usage_cost_layout.setContentsMargins(5,5,5,5)
        main_layout.addWidget(self.usage_cost_area)

        row = 0
        # Input Tokens
        usage_cost_layout.addWidget(QLabel("Input Tokens:"), row, 0)
        self.input_tokens_val = QLabel("0")
        usage_cost_layout.addWidget(self.input_tokens_val, row, 1)
        # Output Tokens
        usage_cost_layout.addWidget(QLabel("Output Tokens:"), row, 2)
        self.output_tokens_val = QLabel("0")
        usage_cost_layout.addWidget(self.output_tokens_val, row, 3)
        row += 1
        # Cache Creation Tokens
        usage_cost_layout.addWidget(QLabel("Cache Create Tokens:"), row, 0)
        self.cache_create_tokens_val = QLabel("0")
        usage_cost_layout.addWidget(self.cache_create_tokens_val, row, 1)
        # Cache Read Tokens
        usage_cost_layout.addWidget(QLabel("Cache Read Tokens:"), row, 2)
        self.cache_read_tokens_val = QLabel("0")
        usage_cost_layout.addWidget(self.cache_read_tokens_val, row, 3)
        row += 1
        # Actual Cost
        usage_cost_layout.addWidget(QLabel("<b>Session Cost:</b>"), row, 0)
        self.actual_cost_val = QLabel("$0.000000")
        usage_cost_layout.addWidget(self.actual_cost_val, row, 1)
        # Hypothetical Cost & Savings
        usage_cost_layout.addWidget(QLabel("Cost w/o Cache:"), row, 2)
        self.hypo_cost_val = QLabel("$0.000000")
        usage_cost_layout.addWidget(self.hypo_cost_val, row, 3)
        self.savings_label = QLabel("Savings: $0.000000 (0.00%)")
        usage_cost_layout.addWidget(self.savings_label, row, 4, 1, 2) # Span 2 columns for savings
        self.savings_label.setVisible(False) # Initially hidden

        # Set column stretches for a more balanced layout in the grid
        usage_cost_layout.setColumnStretch(1, 1)
        usage_cost_layout.setColumnStretch(3, 1)
        usage_cost_layout.setColumnStretch(4, 1)        
        
        # Status bar for usage (optional)
        self.statusBar().showMessage("Initializing chat view...") # New initial message

        # Initially disable input field too
        self.input_field.setEnabled(False)


    def _init_chat_html(self):
        # Initial HTML structure for the QWebEngineView
        # Includes basic styling and a container for messages.
        # MathJax will be loaded by ChatWebEnginePage
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
            <style>
                body { font-family: sans-serif; margin: 0; padding: 10px; background-color: #2E2E2E; color: #F0F0F0; font-size: 11pt; }
                .message-container { display: flex; flex-direction: column; }
                .message { padding: 10px; margin-bottom: 10px; border-radius: 8px; max-width: 85%; word-wrap: break-word; }
                .user-message { background-color: #4A4A6A; align-self: flex-end; margin-left: 15%; }
                .assistant-message { background-color: #3C4C3C; align-self: flex-start; margin-right: 15%; }
                .message img { max-width: 100%; height: auto; border-radius: 5px; margin-top: 5px; }
                .message .file-info { font-size: 0.9em; color: #B0B0B0; margin-top: 5px; }
                .mathjax-processed { color: #F0F0F0 !important; } /* Ensure MathJax output is visible */
                code { background-color: #454545; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
                pre code { display: block; padding: 8px; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <div id="message-container" class="message-container">
                <!-- Messages will be appended here -->
            </div>
            <script>
                console.log('Initial script block loading.');
                function appendMessage(role, htmlContent, messageId) {
                    console.log('[JS] appendMessage called:', 
                                'Role:', role, 
                                'MsgID:', messageId, 
                                'HTML:', htmlContent.substring(0, 100) + '...');
                    try {
                        const container = document.getElementById('message-container');
                        console.log('[JS] appendMessage: message-container element:', container);
                        if (!container) {
                            console.error('[JS] appendMessage: CRITICAL - message-container NOT FOUND!');
                            return;
                        }
                        const msgDiv = document.createElement('div');
                        msgDiv.classList.add('message', role === 'user' ? 'user-message' : 'assistant-message');
                        if (messageId) {
                            msgDiv.id = messageId;
                        }
                        msgDiv.innerHTML = htmlContent;
                        container.appendChild(msgDiv);
                        window.scrollTo(0, document.body.scrollHeight); // Auto-scroll
                        
                        // Apply syntax highlighting to code blocks within the new message
                        msgDiv.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });

                        // Notify MathJax to typeset new content if it's an assistant message
                        if (role === 'assistant' && window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                             window.MathJax.startup.promise.then(() => {
                                MathJax.typesetPromise([msgDiv]).catch(function (err) { console.error('MathJax typesetting error:', err); });
                             });
                        }
                        console.log('[JS] appendMessage: Succeeded for MsgID:', messageId);
                    } catch (e) {
                        console.error('[JS] appendMessage Error:', e, 'Role:', role, 'MsgID:', messageId);
                    }
                }

                function updateMessage(messageId, htmlContent) {
                    console.log('[JS] updateMessage called:', 
                                'MsgID:', messageId, 
                                'HTML:', htmlContent.substring(0,100) + '...');
                    try {
                        const msgDiv = document.getElementById(messageId);
                        console.log('[JS] updateMessage: element for MsgID (' + messageId + '):', msgDiv);
                        if (!msgDiv) {
                            console.error('[JS] updateMessage: CRITICAL - Element with ID ' + messageId + ' NOT FOUND!');
                            return;
                        }
                        msgDiv.innerHTML = htmlContent;
                        window.scrollTo(0, document.body.scrollHeight);

                        // Apply syntax highlighting to code blocks within the updated message
                        msgDiv.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });

                        if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                            window.MathJax.startup.promise.then(() => {
                                MathJax.typesetPromise([msgDiv]).catch(function (err) { console.error('MathJax typesetting error:', err); });
                            });
                        }
                        console.log('[JS] updateMessage: Succeeded for MsgID:', messageId);
                    } catch (e) {
                         console.error('[JS] updateMessage Error:', e, 'MsgID:', messageId);
                    }
                }
                
                // Optional: listen for new messages from Python to trigger MathJax
                // This can be done via document events if qwebchannel is too complex for this.
                // Example: document.dispatchEvent(new CustomEvent('newMessageForMathJax'));
            </script>
        </body>
        </html>
        """
        self.chat_view.setHtml(html_content, QUrl("app://localhost/chat.html"))


    def _apply_styles(self):
        # Global styles for Qt Widgets (QWebEngineView is styled via its HTML/CSS)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2E2E2E;
                color: #F0F0F0;
            }
            QTextEdit {
                background-color: #3C3C3C;
                color: #F0F0F0;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 8px;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #555555;
                color: #F0F0F0;
                border: 1px solid #666666;
                padding: 8px 12px;
                border-radius: 5px;
                font-size: 10pt;
                min-height: 25px; /* Ensure buttons have some height */
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QLabel {
                color: #D0D0D0;
                font-size: 9pt;
                padding-top: 5px;
            }
            QStatusBar {
                color: #D0D0D0;
            }
        """)

    def _handle_input_key_press(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if not (event.modifiers() & Qt.ShiftModifier):
                self._send_message_slot()
                return 
        QTextEdit.keyPressEvent(self.input_field, event)

    @Slot()
    def _attach_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach Files", "",
            "Files (*.png *.jpg *.jpeg *.gif *.webp *.pdf);;Images (*.png *.jpg *.jpeg *.gif *.webp);;PDF (*.pdf);;All Files (*)"
        )
        if file_paths:
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                media_type, _ = mimetypes.guess_type(file_path)
                if not media_type: # Fallback guess
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']: media_type = f"image/{ext[1:]}"
                    elif ext == '.pdf': media_type = "application/pdf"
                    else: media_type = "application/octet-stream"
                
                try:
                    with open(file_path, "rb") as f: file_bytes = f.read()
                    base64_data = base64.b64encode(file_bytes).decode("utf-8")
                    
                    if not any(f['name'] == file_name for f in self.uploaded_files_data):
                        self.uploaded_files_data.append({
                            "name": file_name, "type": media_type,
                            "data": base64_data, "path_for_display": file_path
                        })
                except Exception as e:
                    self._add_message_to_view("error", f"Error attaching {file_name}: {e}")
            self._update_staged_files_display()

    def _update_staged_files_display(self):
        for i in reversed(range(self.staged_files_layout.count())): 
            self.staged_files_layout.itemAt(i).widget().deleteLater()

        if not self.uploaded_files_data:
            self.staged_files_label.setVisible(False)
            self.staged_files_display_area.setVisible(False)
            return

        for file_info in self.uploaded_files_data:
            item_widget = QWidget()
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(0,0,0,0)
            
            icon_label = QLabel()
            icon_label.setAlignment(Qt.AlignCenter)
            if file_info["type"].startswith("image/"):
                pixmap = QPixmap()
                pixmap.loadFromData(base64.b64decode(file_info["data"]))
                icon_label.setPixmap(pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif file_info["type"] == "application/pdf": icon_label.setText("📄") # PDF icon
            else: icon_label.setText("📎") # Generic file icon
            
            name_label = QLabel(file_info["name"][:12] + "..." if len(file_info["name"]) > 12 else file_info["name"])
            name_label.setAlignment(Qt.AlignCenter)
            
            item_layout.addWidget(icon_label)
            item_layout.addWidget(name_label)
            item_widget.setToolTip(file_info["name"])
            self.staged_files_layout.addWidget(item_widget)
        
        self.staged_files_label.setVisible(True)
        self.staged_files_display_area.setVisible(True)

    def _prepare_content_for_api_and_display(self, user_prompt_text):
        api_content_blocks = []
        display_html_parts = []

        for file_info in self.uploaded_files_data:
            if file_info["type"].startswith("image/"):
                api_content_blocks.append({
                    "type": "image", "source": {
                        "type": "base64", "media_type": file_info["type"], "data": file_info["data"]
                    }, "cache_control": {"type": "ephemeral"} # Added cache_control like Streamlit
                })
                display_html_parts.append(f"<img src='data:{file_info['type']};base64,{file_info['data']}' alt='{file_info['name']}'/><div class='file-info'>{file_info['name']}</div>")
            elif file_info["type"] == "application/pdf":
                api_content_blocks.append({
                    "type": "document", "source": { 
                        "type": "base64", "media_type": "application/pdf", "data": file_info["data"]
                        # Removed "name": file_info["name"] from here
                    }, "cache_control": {"type": "ephemeral"} # Added cache_control
                })
                display_html_parts.append(f"<div class='file-info'>📄 PDF Uploaded: {file_info['name']} (sent to AI)</div>")
            else:
                display_html_parts.append(f"<div class='file-info'>📎 File: {file_info['name']} (type: {file_info['type']}) - May not be processed by AI.</div>")

        if user_prompt_text:
            api_content_blocks.append({"type": "text", "text": user_prompt_text})
            display_html_parts.append(f"<div>{self.markdown_parser(user_prompt_text)}</div>")
        
        # Anthropic requires a text block if other content types are present.
        if any(item["type"] != "text" for item in api_content_blocks) and not any(item["type"] == "text" for item in api_content_blocks):
            api_content_blocks.append({"type": "text", "text": "Analyze the uploaded content."})
            if not user_prompt_text: # Add placeholder if user only uploaded files
                display_html_parts.append("<div><small>[AI prompted to analyze uploaded content]</small></div>")

        return api_content_blocks, "".join(display_html_parts)

    @Slot()
    def _send_message_slot(self):
        user_prompt = self.input_field.toPlainText().strip()
        if not user_prompt and not self.uploaded_files_data:
            self._show_error_popup("Please enter a message or attach a file.")
            return

        if not self.initial_page_loaded:
            self.statusBar().showMessage("Chat view is not ready yet. Please wait...", 3000)
            return

        self._set_ui_busy(True)

        api_blocks, display_html = self._prepare_content_for_api_and_display(user_prompt)
        self._add_message_to_view("user", display_html)
        self.chat_history_api.append({"role": "user", "content": api_blocks})

        self.input_field.clear()
        self.uploaded_files_data = []
        self._update_staged_files_display()

        self.current_assistant_message_id = f"assistant_msg_{QDateTime.currentMSecsSinceEpoch()}"
        self._add_message_to_view("assistant", "Thinking...", self.current_assistant_message_id)
        
        # Create new worker for each call
        if self.anthropic_worker and self.anthropic_worker.isRunning(): # Should not happen if UI is disabled
            self.anthropic_worker.stop()
            self.anthropic_worker.wait()

        self.anthropic_worker = AnthropicWorker(
            ANTHROPIC_API_KEY, MODEL_NAME, list(self.chat_history_api), MAX_TOKENS_OUTPUT # Send a copy
        )
        self.anthropic_worker.response_chunk.connect(self._handle_response_chunk)
        self.anthropic_worker.response_error.connect(self._handle_response_error)
        self.anthropic_worker.response_finished.connect(self._handle_response_finished)
        self.anthropic_worker.start()

    def _set_ui_busy(self, busy):
        self.input_field.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.attach_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.reset_button.setEnabled(busy)

    def _add_message_to_view(self, role, html_content, message_id=None):
        # Use json.dumps to safely escape html_content for JavaScript
        escaped_html_json = json.dumps(html_content)
        
        js_call = ""
        if role == "assistant" and message_id:
            if message_id not in self.created_assistant_message_ids:
                # First time for this assistant message ID, so append it.
                js_call = f"appendMessage('{role}', {escaped_html_json}, '{message_id}');"
                self.created_assistant_message_ids.add(message_id)
            else:
                # Assistant message div already exists, so update it.
                js_call = f"updateMessage('{message_id}', {escaped_html_json});"
        else: # User messages or assistant messages without a specific ID (should not happen for streaming)
            message_id_arg = f"'{message_id}'" if message_id else "null"
            js_call = f"appendMessage('{role}', {escaped_html_json}, {message_id_arg});"
            if role == "assistant" and message_id: # Should be caught above, but for safety
                 self.created_assistant_message_ids.add(message_id)
        
        print(f"[Python] Attempting to run JS: {js_call[:250]} ...") # Log the JS call
        self.chat_page.runJavaScript(js_call)

    @Slot(str)
    def _handle_response_chunk(self, chunk):
        if self.current_assistant_message_id:
            current_content_js = f"document.getElementById('{self.current_assistant_message_id}').innerHTML;"
            
            def callback(current_html):
                # Remove "Thinking..." if it's there
                if current_html == "Thinking...": current_html = ""
                
                # Append new chunk (ensure it's treated as text, then parsed by Markdown)
                # This is simplified; for true Markdown streaming, parsing needs to be incremental or deferred.
                # For now, accumulate text and re-render Markdown.
                
                # Let's assume chunk is plain text, combine and then Markdown parse
                # To do this properly, need to store raw text from chunks for assistant
                if not hasattr(self, 'current_assistant_raw_text'):
                    self.current_assistant_raw_text = ""
                self.current_assistant_raw_text += chunk
                
                # Process accumulated raw text with Markdown
                processed_html = self.markdown_parser(self.current_assistant_raw_text)
                self._add_message_to_view("assistant", processed_html, self.current_assistant_message_id)

            self.chat_page.runJavaScript(current_content_js, callback)


    @Slot(str)
    def _handle_response_error(self, error_message):
        if self.current_assistant_message_id:
            self._add_message_to_view("assistant", f"<font color='red'>Error: {error_message}</font>", self.current_assistant_message_id)
        else: # Should not happen if error is for a message in progress
            self._add_message_to_view("error", f"<font color='red'>Error: {error_message}</font>")
        
        self.statusBar().showMessage(f"Error: {error_message[:100]}...", 5000)
        self._set_ui_busy(False)
        self.current_assistant_message_id = None
        if hasattr(self, 'current_assistant_raw_text'): del self.current_assistant_raw_text


    @Slot(dict)
    def _handle_response_finished(self, result_data):
        # Final update with fully accumulated and processed content
        final_text = ""
        if hasattr(self, 'current_assistant_raw_text'):
            final_text = self.current_assistant_raw_text
            del self.current_assistant_raw_text # Clean up

        usage_info_dict = result_data.get("usage", {})
        stop_reason = result_data.get("stop_reason", "unknown")

        # Final processing of accumulated text
        final_html_content = self.markdown_parser(final_text)
        
        if self.current_assistant_message_id:
            self._add_message_to_view("assistant", final_html_content, self.current_assistant_message_id)
        
        self.chat_history_api.append({"role": "assistant", "content": [{"type": "text", "text": final_text}]}) # Store raw model output
        
        current_turn_cost = 0.0
        current_turn_hypothetical_cost = 0.0
        model_costs_for_turn = self._get_model_costs(MODEL_NAME)

        if usage_info_dict: # If we got usage data
            token_types_for_costing = [
                "input_tokens", "output_tokens", 
                "cache_creation_input_tokens", "cache_read_input_tokens"
            ]
            
            for token_type_api_name in token_types_for_costing:
                tokens_used_in_type = usage_info_dict.get(token_type_api_name, 0)
                cost_per_mil = model_costs_for_turn.get(token_type_api_name, 0) # Get specific cost for this token type
                
                if tokens_used_in_type > 0:
                    # Accumulate to session_total_usage
                    self.session_total_usage[token_type_api_name] = \
                        self.session_total_usage.get(token_type_api_name, 0) + tokens_used_in_type
                    
                    # Calculate actual cost for this token type
                    current_turn_cost += (tokens_used_in_type / 1_000_000) * cost_per_mil

            self.session_total_cost += current_turn_cost

            # Calculate hypothetical cost for the turn
            # Cost of (input_tokens + cache_creation_input_tokens) at standard input_tokens rate
            hypo_input_standard_cost_rate = model_costs_for_turn.get("input_tokens", 0)
            hypo_output_standard_cost_rate = model_costs_for_turn.get("output_tokens", 0)

            # Actual input tokens + tokens used for creating cache (these are effectively inputs)
            effective_input_tokens = usage_info_dict.get("input_tokens", 0) + \
                                     usage_info_dict.get("cache_creation_input_tokens", 0)
            current_turn_hypothetical_cost += (effective_input_tokens / 1_000_000) * hypo_input_standard_cost_rate

            # Actual output tokens
            output_tokens = usage_info_dict.get("output_tokens", 0)
            current_turn_hypothetical_cost += (output_tokens / 1_000_000) * hypo_output_standard_cost_rate

            # If cache_read_input_tokens were used, for hypo, cost them as if they were fresh input_tokens
            cache_read_tokens = usage_info_dict.get("cache_read_input_tokens", 0)
            current_turn_hypothetical_cost += (cache_read_tokens / 1_000_000) * hypo_input_standard_cost_rate
            
            self.session_total_hypothetical_cost += current_turn_hypothetical_cost

        # Display status bar message
        if usage_info_dict:
            input_tk = usage_info_dict.get('input_tokens', 0)
            output_tk = usage_info_dict.get('output_tokens', 0)
            self.statusBar().showMessage(f"Finished. Input: {input_tk} tokens, Output: {output_tk} tokens. Stop reason: {stop_reason}", 10000)
        else:
            self.statusBar().showMessage(f"Finished. Stop reason: {stop_reason}", 5000)

        self._set_ui_busy(False)
        self.current_assistant_message_id = None
        self._update_usage_display() # Update display after response

    @Slot()
    def _stop_generation(self):
        if self.anthropic_worker and self.anthropic_worker.isRunning():
            self.anthropic_worker.stop()
            self.statusBar().showMessage("Stop request sent. Waiting for current operation to halt...", 3000)
            # UI will be re-enabled by error or finished signal handler
            self.stop_button.setEnabled(False) # Prevent multiple clicks

    def _show_error_popup(self, message):
        QMessageBox.critical(self, "Error", message)
        
    def _get_model_costs(self, model_name_to_check):
        """Safely retrieves costs for a given model, falling back to defaults."""
        return COSTS_PER_MILLION_TOKENS.get(model_name_to_check, DEFAULT_MODEL_COSTS)

    @Slot(bool)
    def _handle_initial_load_finished(self, success):
        print(f"[Python] _handle_initial_load_finished called. Success: {success}") # Log this event
        if success:
            self.initial_page_loaded = True
            if ANTHROPIC_API_KEY:
                self.input_field.setEnabled(True)
                self.send_button.setEnabled(True)
                self.attach_button.setEnabled(True)
                self.reset_button.setEnabled(True) # Enable reset button
                self.statusBar().showMessage("Ready.", 3000)
            else:
                self.statusBar().showMessage("ANTHROPIC_API_KEY is missing. Please set it in .env", 5000)
                self._show_error_popup("ANTHROPIC_API_KEY not found. Chat functionality will be limited.")
        else:
            self.statusBar().showMessage("Error: Chat display failed to load.")
            self._show_error_popup("Critical Error: Chat display (QWebEngineView) failed to load. The application may not function correctly.")

    def closeEvent(self, event):
        if self.anthropic_worker and self.anthropic_worker.isRunning():
            self.anthropic_worker.stop()
            if not self.anthropic_worker.wait(3000): # Wait up to 3s
                 print("Anthropic worker did not terminate gracefully.")
        # Clean up WebEngineView properly
        self.chat_view.page().profile().clearHttpCache() # Optional: clear cache
        self.chat_view.stop()
        self.chat_view.close() # Explicitly close web view
        del self.chat_view # Ensure it's deleted
        super().closeEvent(event)

    @Slot()
    def _update_usage_display(self):
        # This will be filled in later to update the QLabel texts
        # For now, just a print to confirm it's called
        print("[Python] _update_usage_display called")
        self.input_tokens_val.setText(f"{self.session_total_usage.get('input_tokens', 0):,}")
        self.output_tokens_val.setText(f"{self.session_total_usage.get('output_tokens', 0):,}")
        self.cache_create_tokens_val.setText(f"{self.session_total_usage.get('cache_creation_input_tokens', 0):,}")
        self.cache_read_tokens_val.setText(f"{self.session_total_usage.get('cache_read_input_tokens', 0):,}")

        self.actual_cost_val.setText(f"${self.session_total_cost:.6f}")
        
        if self.session_total_hypothetical_cost > self.session_total_cost and self.session_total_hypothetical_cost > 0:
            savings = self.session_total_hypothetical_cost - self.session_total_cost
            savings_percentage = (savings / self.session_total_hypothetical_cost) * 100
            self.hypo_cost_val.setText(f"${self.session_total_hypothetical_cost:.6f}")
            self.savings_label.setText(f"Savings: ${savings:.6f} ({savings_percentage:.2f}%)")
            self.savings_label.setVisible(True)
        else:
            self.hypo_cost_val.setText(f"${self.session_total_hypothetical_cost:.6f}") # Show hypo cost even if no savings
            self.savings_label.setVisible(False)

    @Slot()
    def _reset_chat_slot(self):
        # This will be filled in later
        # For now, just a print to confirm it's called
        print("[Python] _reset_chat_slot called")

        # Stop any current generation
        if self.anthropic_worker and self.anthropic_worker.isRunning():
            self._stop_generation()
            # Consider waiting for worker to finish or forcibly terminate if necessary

        self.chat_history_api = []
        self.uploaded_files_data = []
        self._update_staged_files_display() # Clear staged files UI
        self.created_assistant_message_ids = set()
        
        # Reset statistics
        self.session_total_usage = {}
        self.session_total_cost = 0.0
        self.session_total_hypothetical_cost = 0.0
        
        self._init_chat_html() # Reloads the initial empty chat HTML
        self._update_usage_display() # Update display to show zeros
        self.statusBar().showMessage("Chat reset.", 3000)


if __name__ == "__main__":
    # Required for QWebEngineView in some environments
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) # Deprecated in Qt6
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps) # Deprecated in Qt6

    app = QApplication(sys.argv)
    
    # Basic Dark Theme Palette (can be customized further with QSS)
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(46, 46, 46))
    dark_palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Base, QColor(60, 60, 60))
    dark_palette.setColor(QPalette.AlternateBase, QColor(70, 70, 70))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(46, 46, 46))
    dark_palette.setColor(QPalette.ToolTipText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Text, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Button, QColor(85, 85, 85))
    dark_palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    app.setPalette(dark_palette)
    app.setStyle("Fusion") # Fusion style often works well with palettes

    window = ChatWindow()
    window.show()
    sys.exit(app.exec()) 