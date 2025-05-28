document.addEventListener("DOMContentLoaded", function () {
    const chatMessages = document.getElementById("chatMessages");
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendButton");
    const stopButton = document.getElementById("stopButton");
    const modelNameSpan = document.getElementById("modelName");
    const totalTokenUsageDiv = document.getElementById("totalTokenUsage");
    const totalSessionCostDiv = document.getElementById("totalSessionCost");

    let websocket;
    let currentAssistantMessageElement = null;
    let currentAssistantMessageDetailsElement = null;
    let streamingInProgress = false;

    let sessionTotalUsage = {};
    let sessionTotalCost = 0.0;
    let sessionTotalHypotheticalCost = 0.0;

    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        websocket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        websocket.onopen = function (event) {
            console.log("WebSocket connection established");
            // You could request initial model name here if needed, or assume backend sends it.
        };

        websocket.onmessage = function (event) {
            const data = JSON.parse(event.data);
            console.log("Received from server:", data);

            if (data.type === "text_chunk") {
                if (!currentAssistantMessageElement) {
                    // This should not happen if message display logic is correct
                    console.error("currentAssistantMessageElement is null when receiving chunk");
                    appendMessage("assistant", ""); // Create one as a fallback
                }
                // Append text to the content part of the assistant message
                const contentElement = currentAssistantMessageElement.querySelector('.message-content');
                contentElement.textContent += data.content;
                chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
            } else if (data.type === "stream_status") {
                streamingInProgress = false;
                stopButton.style.display = "none";
                sendButton.disabled = false;
                messageInput.disabled = false;

                if (data.status === "stopped_by_user") {
                    if (currentAssistantMessageElement) {
                         const contentElement = currentAssistantMessageElement.querySelector('.message-content');
                        if (!contentElement.textContent.trim()) {
                            contentElement.textContent = "[INFO] Streaming was stopped by user before any content was generated.";
                        } else {
                            contentElement.textContent += "\n\n[INFO] Streaming stopped by user.";
                        }
                        if (!currentAssistantMessageDetailsElement) {
                             currentAssistantMessageDetailsElement = document.createElement("div");
                             currentAssistantMessageDetailsElement.className = "message-details";
                             currentAssistantMessageElement.appendChild(currentAssistantMessageDetailsElement);
                        }
                        currentAssistantMessageDetailsElement.textContent = "[INFO] Streaming was stopped by the user. Full token usage for this partial response is unavailable.";
                    }
                }
                // Reset for the next message
                currentAssistantMessageElement = null;
                currentAssistantMessageDetailsElement = null;
            } else if (data.type === "usage_update") {
                modelNameSpan.textContent = data.model_name;
                updateTurnStats(data.usage, data.cost, data.model_name);
                updateSessionTotals(data.usage, data.cost, data.hypothetical_cost);
            } else if (data.type === "error") {
                appendMessage("system", `Error: ${data.message}`, "error");
                streamingInProgress = false;
                stopButton.style.display = "none";
                sendButton.disabled = false;
                messageInput.disabled = false;
                currentAssistantMessageElement = null; 
                currentAssistantMessageDetailsElement = null;
            } else if (data.type === "stop_ack") {
                console.log("Server acknowledged stop request.");
                // UI might show a message like "Stopping..."
            } else if (data.type === "stream_end") {
                console.log("Stream processing finished on server.");
                // Final UI updates post-stream if any
                sendButton.disabled = false;
                messageInput.disabled = false;
                stopButton.style.display = "none";
                streamingInProgress = false;
            }
        };

        websocket.onclose = function (event) {
            console.log("WebSocket connection closed. Attempting to reconnect...");
            streamingInProgress = false;
            stopButton.style.display = "none";
            sendButton.disabled = true;
            messageInput.disabled = true;
            appendMessage("system", "Connection lost. Attempting to reconnect...", "error");
            setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
        };

        websocket.onerror = function (error) {
            console.error("WebSocket error:", error);
            streamingInProgress = false;
            stopButton.style.display = "none";
            sendButton.disabled = false; // Or true, depending on desired behavior
            messageInput.disabled = false;
            appendMessage("system", "WebSocket connection error.", "error");
            // Reconnection is handled by onclose
        };
    }

    function appendMessage(role, text, type = "normal") {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${role}-message`);
        if (type === "error") {
            messageElement.classList.add("error-message");
        }
        
        const contentElement = document.createElement("div");
        contentElement.className = "message-content";
        contentElement.textContent = text;
        messageElement.appendChild(contentElement);

        chatMessages.appendChild(messageElement);

        if (role === "assistant") {
            currentAssistantMessageElement = messageElement;
            // Detail element will be added when usage_update comes or if stopped
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageElement; // Return the top-level message element
    }

    function updateTurnStats(usage, cost, modelName) {
        if (!currentAssistantMessageElement) {
            console.error("No current assistant message element to update stats for.");
            return;
        }
        // Ensure detail element exists
        if (!currentAssistantMessageDetailsElement) {
            currentAssistantMessageDetailsElement = document.createElement("div");
            currentAssistantMessageDetailsElement.className = "message-details";
            currentAssistantMessageElement.appendChild(currentAssistantMessageDetailsElement);
        }

        let details = [];
        const tokenTypesOrdered = [
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ];

        tokenTypesOrdered.forEach(tokenType => {
            const count = usage[tokenType] || 0;
            if (count > 0) {
                details.push(`${tokenType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: ${count}`);
            }
        });

        if (cost > 0) {
            details.push(`Cost: $${cost.toFixed(6)}`);
        }

        currentAssistantMessageDetailsElement.textContent = details.join(", ");
    }

    function updateSessionTotals(turnUsage, turnCost, turnHypotheticalCost) {
        for (const tokenType in turnUsage) {
            if (turnUsage[tokenType] > 0) {
                sessionTotalUsage[tokenType] = (sessionTotalUsage[tokenType] || 0) + turnUsage[tokenType];
            }
        }
        sessionTotalCost += turnCost;
        sessionTotalHypotheticalCost += turnHypotheticalCost;

        // Update UI for total token usage
        totalTokenUsageDiv.innerHTML = ""; // Clear previous
        let hasUsage = false;
        for (const tokenType in sessionTotalUsage) {
            if (sessionTotalUsage[tokenType] > 0) {
                hasUsage = true;
                const p = document.createElement("p");
                p.textContent = `Total ${tokenType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: ${sessionTotalUsage[tokenType]}`;
                totalTokenUsageDiv.appendChild(p);
            }
        }
        if (!hasUsage) {
            totalTokenUsageDiv.innerHTML = "<p>No API calls yet in this session.</p>";
        }

        // Update UI for total session cost
        let costLabel = "Session Cost";
        if (sessionTotalHypotheticalCost > 0 && sessionTotalHypotheticalCost > sessionTotalCost) {
            const savings = sessionTotalHypotheticalCost - sessionTotalCost;
            const savingsPercentage = (savings / sessionTotalHypotheticalCost) * 100;
            costLabel += ` (Saved ${savingsPercentage.toFixed(2)}%)`;
        }
        totalSessionCostDiv.innerHTML = `<p>${costLabel}: $${sessionTotalCost.toFixed(6)}</p>`;
    }


    function sendMessage() {
        const messageText = messageInput.value.trim();
        if (messageText === "") return;

        if (streamingInProgress && websocket && websocket.readyState === WebSocket.OPEN) {
            // If already streaming, first send a stop signal
            websocket.send(JSON.stringify({ action: "stop_streaming" }));
            // Optionally, give a visual cue or wait for stop_ack before proceeding.
            // For simplicity here, we'll just proceed to send new message, 
            // server side handles new message by stopping old stream.
        }

        appendMessage("user", messageText);
        messageInput.value = "";

        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ action: "send_message", message: messageText }));
            streamingInProgress = true;
            sendButton.disabled = true;
            messageInput.disabled = true;
            stopButton.style.display = "block";
            
            // Create placeholder for assistant's response immediately
            currentAssistantMessageElement = appendMessage("assistant", "");
            currentAssistantMessageDetailsElement = document.createElement("div");
            currentAssistantMessageDetailsElement.className = "message-details";
            // Do not append currentAssistantMessageDetailsElement yet, wait for usage data or stop signal
        } else {
            appendMessage("system", "WebSocket is not connected. Please wait or refresh.", "error");
        }
    }

    sendButton.addEventListener("click", sendMessage);
    messageInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    stopButton.addEventListener("click", function () {
        if (streamingInProgress && websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ action: "stop_streaming" }));
            stopButton.disabled = true; // Prevent multiple clicks
            // UI can show "Stopping..." message
            // The server will send stream_status: stopped_by_user and then stream_end
            // which will re-enable buttons.
        }
    });

    // Initial connection
    connectWebSocket();
}); 