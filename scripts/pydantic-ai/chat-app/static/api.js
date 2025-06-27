export async function fetchSessions() {
  const res = await fetch('/sessions/');
  return res.json();
}

export async function createSession() {
  const res = await fetch('/session/', { method: 'POST' });
  return res.json();
}

export async function fetchMessages(sessionId) {
  const res = await fetch(`/chat/?session_id=${encodeURIComponent(sessionId)}`);
  const text = await res.text();
  return text
    .split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => JSON.parse(line));
}

export async function sendMessage(sessionId, prompt) {
  const body = new FormData();
  body.append('session_id', sessionId);
  body.append('prompt', prompt);
  const res = await fetch('/chat/', { method: 'POST', body });
  const text = await res.text();
  return text
    .split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => JSON.parse(line));
} 