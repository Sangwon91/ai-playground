import { fetchSessions, createSession, fetchMessages } from './api.js';
import { getCurrentSession, setCurrentSession } from './state.js';
import { renderSessionList, renderMessages, showSpinner, showError, clearConversation, setCurrentSessionUI } from './ui.js';

async function loadAndRenderSessions() {
  const sessions = await fetchSessions();
  renderSessionList(sessions, getCurrentSession(), handleSessionClick);
}

async function loadAndRenderMessages() {
  showSpinner(true);
  try {
    const messages = await fetchMessages(getCurrentSession());
    renderMessages(messages);
  } catch (e) {
    showError('메시지 불러오기 실패');
  }
  showSpinner(false);
}

function handleSessionClick(sessionId) {
  setCurrentSession(sessionId);
  setCurrentSessionUI(sessionId);
  clearConversation();
  loadAndRenderMessages();
  loadAndRenderSessions();
}

async function handleNewSession() {
  const data = await createSession();
  handleSessionClick(data.session_id);
}

async function handleFormSubmit(e) {
  e.preventDefault();
  showSpinner(true);
  const promptInput = document.getElementById('prompt-input');
  const prompt = promptInput.value;
  promptInput.value = '';
  promptInput.disabled = true;
  try {
    // StreamingResponse를 직접 읽어서 실시간 렌더링
    const body = new FormData();
    body.append('session_id', getCurrentSession());
    body.append('prompt', prompt);
    const response = await fetch('/chat/', { method: 'POST', body });
    if (!response.body) throw new Error('No response body');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    clearConversation();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (line.trim().length > 0) {
          renderMessages([JSON.parse(line)]); // 한 줄씩 렌더링
        }
      }
    }
    if (buffer.trim().length > 0) {
      renderMessages([JSON.parse(buffer)]);
    }
    await loadAndRenderSessions();
  } catch (e) {
    showError('메시지 전송 실패');
  }
  promptInput.disabled = false;
  showSpinner(false);
}

document.addEventListener('DOMContentLoaded', () => {
  setCurrentSession('default');
  setCurrentSessionUI('default');
  loadAndRenderSessions();
  loadAndRenderMessages();

  document.getElementById('new-session-btn').onclick = handleNewSession;
  document.querySelector('form').addEventListener('submit', handleFormSubmit);
}); 