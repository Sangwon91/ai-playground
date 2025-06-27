import { renderMathWithRetry } from './math.js';

export function renderSessionList(sessions, currentSessionId, onClick) {
  const sessionList = document.getElementById('session-list');
  sessionList.innerHTML = '';
  for (const s of sessions) {
    const li = document.createElement('li');
    li.className = `rounded-lg border bg-card px-4 py-2 shadow-sm cursor-pointer flex flex-col gap-1 transition \
      ${s.session_id === currentSessionId ? 'ring-2 ring-primary bg-accent/40 border-primary font-bold' : 'hover:bg-accent/20'}`;
    li.style.background = s.session_id === currentSessionId ? 'var(--accent)' : 'var(--card)';
    li.style.borderColor = s.session_id === currentSessionId ? 'var(--primary)' : 'var(--border)';
    li.innerHTML = `
      <div class="font-mono text-xs truncate" style="color: var(--muted-foreground);">${s.session_id}</div>
      <div class="truncate text-sm">${s.last_message ? s.last_message.slice(0, 40) : ''}</div>
      <div class="text-xs" style="color: var(--muted-foreground);">${s.last_time ? s.last_time.slice(0,19).replace('T',' ') : ''}</div>
    `;
    li.onclick = () => onClick(s.session_id);
    sessionList.appendChild(li);
  }
}

export function renderMessages(messages) {
  const convElement = document.getElementById('conversation');
  convElement.innerHTML = '';
  for (const message of messages) {
    const { timestamp, role, content } = message;
    const id = `msg-${timestamp}`;
    let msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.title = `${role} at ${timestamp}`;
    msgDiv.classList.add('flex', 'w-full', 'mb-2');
    msgDiv.classList.add(role === 'user' ? 'justify-start' : 'justify-end');
    const bubble =
      role === 'user'
        ? `<div class="max-w-[70%] rounded-xl rounded-bl-none bg-muted text-foreground p-4 shadow border" style="background: var(--muted); color: var(--foreground); border-color: var(--border);">
            <div class="text-xs mb-1 text-muted-foreground">You</div>
            <div class="prose break-words">${window.marked.parse(content)}</div>
          </div>`
        : `<div class="max-w-[70%] rounded-xl rounded-br-none bg-primary text-primary-foreground p-4 shadow border" style="background: var(--primary); color: var(--primary-foreground); border-color: var(--primary);">
            <div class="text-xs mb-1 text-primary-foreground/80">AI</div>
            <div class="prose break-words">${window.marked.parse(content)}</div>
          </div>`;
    msgDiv.innerHTML = bubble;
    convElement.appendChild(msgDiv);
    const prose = msgDiv.querySelector('.prose');
    if (prose) renderMathWithRetry(prose);
  }
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

export function showSpinner(show) {
  const spinner = document.getElementById('spinner');
  if (show) spinner.classList.remove('hidden');
  else spinner.classList.add('hidden');
}

export function showError(msg) {
  const errorDiv = document.getElementById('error');
  errorDiv.textContent = msg;
  errorDiv.classList.remove('hidden');
}

export function clearConversation() {
  document.getElementById('conversation').innerHTML = '';
}

export function setCurrentSessionUI(sessionId) {
  document.getElementById('current-session').textContent = sessionId;
} 