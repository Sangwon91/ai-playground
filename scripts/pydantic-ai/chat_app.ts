// BIG FAT WARNING: to avoid the complexity of npm, this typescript is compiled in the browser
// there's currently no static type checking

import { marked } from 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'
const convElement = document.getElementById('conversation')

const promptInput = document.getElementById('prompt-input') as HTMLInputElement
const spinner = document.getElementById('spinner')

let currentSessionId = 'default'
const currentSessionSpan = document.getElementById('current-session') as HTMLSpanElement
const sessionList = document.getElementById('session-list') as HTMLUListElement
const newSessionBtn = document.getElementById('new-session-btn') as HTMLButtonElement

function setSession(sessionId: string) {
  currentSessionId = sessionId
  if (currentSessionSpan) currentSessionSpan.textContent = sessionId
  clearConversation()
  loadMessages()
}

function clearConversation() {
  if (convElement) convElement.innerHTML = ''
}

async function loadMessages() {
  try {
    const response = await fetch(`/chat/?session_id=${encodeURIComponent(currentSessionId)}`)
    await onFetchResponse(response)
  } catch (e) {
    onError(e)
  }
}

async function loadSessions() {
  const res = await fetch('/sessions/')
  const sessions = await res.json()
  sessionList.innerHTML = ''
  for (const s of sessions) {
    const li = document.createElement('li')
    li.className = `rounded-lg border bg-card px-4 py-2 shadow-sm cursor-pointer flex flex-col gap-1 transition \
      ${s.session_id === currentSessionId ? 'ring-2 ring-primary bg-accent/40 border-primary font-bold' : 'hover:bg-accent/20'}`
    li.style.background = s.session_id === currentSessionId ? 'var(--accent)' : 'var(--card)'
    li.style.borderColor = s.session_id === currentSessionId ? 'var(--primary)' : 'var(--border)'
    li.innerHTML = `
      <div class="font-mono text-xs truncate" style="color: var(--muted-foreground);">${s.session_id}</div>
      <div class="truncate text-sm">${s.last_message ? s.last_message.slice(0, 40) : ''}</div>
      <div class="text-xs" style="color: var(--muted-foreground);">${s.last_time ? s.last_time.slice(0,19).replace('T',' ') : ''}</div>
    `
    li.onclick = () => setSession(s.session_id)
    sessionList.appendChild(li)
  }
}

if (newSessionBtn) {
  newSessionBtn.onclick = async () => {
    const res = await fetch('/session/', {method: 'POST'})
    const data = await res.json()
    setSession(data.session_id)
    await loadSessions()
  }
}

// stream the response and render messages as each chunk is received
// data is sent as newline-delimited JSON
async function onFetchResponse(response: Response): Promise<void> {
  let text = ''
  let decoder = new TextDecoder()
  if (response.ok) {
    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No reader')
    }
    while (true) {
      const {done, value} = await reader.read()
      if (done) {
        break
      }
      text += decoder.decode(value)
      addMessages(text)
      spinner?.classList.remove('active')
    }
    addMessages(text)
    promptInput.disabled = false
    promptInput.focus()
  } else {
    const text = await response.text()
    console.error(`Unexpected response: ${response.status}`, {response, text})
    throw new Error(`Unexpected response: ${response.status}`)
  }
}

// The format of messages, this matches pydantic-ai both for brevity and understanding
// in production, you might not want to keep this format all the way to the frontend
interface Message {
  role: string
  content: string
  timestamp: string
}

// take raw response text and render messages into the `#conversation` element
// Message timestamp is assumed to be a unique identifier of a message, and is used to deduplicate
// hence you can send data about the same message multiple times, and it will be updated
// instead of creating a new message elements
function addMessages(responseText: string) {
  const lines = responseText.split('\n')
  const messages: Message[] = lines.filter(line => line.length > 1).map(j => JSON.parse(j))
  for (const message of messages) {
    const {timestamp, role, content} = message
    const id = `msg-${timestamp}`
    let msgDiv = document.getElementById(id)
    if (!msgDiv) {
      msgDiv = document.createElement('div')
      msgDiv.id = id
      msgDiv.title = `${role} at ${timestamp}`
      msgDiv.classList.add('flex', 'w-full', 'mb-2')
      if (role === 'user') {
        msgDiv.classList.add('justify-start')
        msgDiv.innerHTML = `
          <div class="max-w-[70%] rounded-xl rounded-bl-none bg-muted text-foreground p-4 shadow border" style="background: var(--muted); color: var(--foreground); border-color: var(--border);">
            <div class="text-xs mb-1 text-muted-foreground">You</div>
            <div class="prose break-words">${marked.parse(content)}</div>
          </div>
        `
      } else {
        msgDiv.classList.add('justify-end')
        msgDiv.innerHTML = `
          <div class="max-w-[70%] rounded-xl rounded-br-none bg-primary text-primary-foreground p-4 shadow border" style="background: var(--primary); color: var(--primary-foreground); border-color: var(--primary);">
            <div class="text-xs mb-1 text-primary-foreground/80">AI</div>
            <div class="prose break-words">${marked.parse(content)}</div>
          </div>
        `
      }
      convElement?.appendChild(msgDiv)
    }
  }
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
}

function renderMathWithRetry(element: HTMLElement, retries = 10) {
  if (window.renderMathInElement) {
    window.renderMathInElement(element, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "$", right: "$", display: false}
      ],
      ignoredTags: []
    })
  } else if (retries > 0) {
    setTimeout(() => renderMathWithRetry(element, retries - 1), 200)
  }
}

function onError(error: any) {
  console.error(error)
  document.getElementById('error')?.classList.remove('d-none')
  document.getElementById('spinner')?.classList.remove('active')
}

// 메시지 전송 시 session_id 포함
async function onSubmit(e: SubmitEvent): Promise<void> {
  e.preventDefault()
  spinner?.classList.add('active')
  const body = new FormData(e.target as HTMLFormElement)
  body.append('session_id', currentSessionId)
  promptInput.value = ''
  promptInput.disabled = true
  const response = await fetch('/chat/', {method: 'POST', body})
  await onFetchResponse(response)
  await loadSessions()
}

document.querySelector('form')?.addEventListener('submit', (e) => onSubmit(e).catch(onError))

// 페이지 로드 시
setSession('default')
loadSessions()

declare global {
  interface Window {
    renderMathInElement?: (el: HTMLElement, options?: any) => void;
  }
}