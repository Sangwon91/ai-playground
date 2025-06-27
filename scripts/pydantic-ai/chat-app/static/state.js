let currentSessionId = 'default';

export function setCurrentSession(id) {
  currentSessionId = id;
}
export function getCurrentSession() {
  return currentSessionId;
} 