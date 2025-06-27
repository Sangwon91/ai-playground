import { fetchSessions, createSession, fetchMessages, sendMessage } from './api.js';
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
    await sendMessage(getCurrentSession(), prompt);
    await loadAndRenderMessages();
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