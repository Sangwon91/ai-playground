// BIG FAT WARNING: to avoid the complexity of npm, this typescript is compiled in the browser
// there's currently no static type checking

import { marked } from 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'
const convElement = document.getElementById('conversation')

const promptInput = document.getElementById('prompt-input') as HTMLInputElement
const spinner = document.getElementById('spinner')

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
      msgDiv.classList.add('border-top', 'pt-2', role)
      convElement?.appendChild(msgDiv)
    }
    msgDiv.innerHTML = marked.parse(content)
    renderMathWithRetry(msgDiv)
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

async function onSubmit(e: SubmitEvent): Promise<void> {
  e.preventDefault()
  spinner?.classList.add('active')
  const body = new FormData(e.target as HTMLFormElement)

  promptInput.value = ''
  promptInput.disabled = true

  const response = await fetch('/chat/', {method: 'POST', body})
  await onFetchResponse(response)
}

// call onSubmit when the form is submitted (e.g. user clicks the send button or hits Enter)
document.querySelector('form')?.addEventListener('submit', (e) => onSubmit(e).catch(onError))

// load messages on page load
fetch('/chat/').then(onFetchResponse).catch(onError)

declare global {
  interface Window {
    renderMathInElement?: (el: HTMLElement, options?: any) => void;
  }
}