// src/components/TravelAgent.jsx
import { useState, useRef, useEffect } from 'react'

const SUGGESTIONS = [
  "Starting at Ballantine now, how do I get to Luddy Hall?",
  "Which bus goes from IMU to the law school?",
  "What's the least crowded bus right now?",
  "Is the route 6 running on time?",
]

// Safely coerce anything to a displayable string
const toStr = (v) => {
  if (typeof v === 'string') return v
  if (v == null) return ''
  try { return JSON.stringify(v) } catch { return String(v) }
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  const text = toStr(msg.content)
  return (
    <div className={`ta-msg ${isUser ? 'ta-msg--user' : 'ta-msg--bot'}`}>
      {!isUser && <div className="ta-msg__avatar">🗺️</div>}
      <div className="ta-msg__bubble">
        {text.split('\n').map((line, i, arr) => (
          <span key={i}>{line}{i < arr.length - 1 && <br />}</span>
        ))}
      </div>
    </div>
  )
}

export default function TravelAgent() {
  const [open, setOpen]         = useState(false)
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [modelUsed, setModelUsed] = useState('')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm your IU transit agent 🗺️ Tell me where you're starting from and where you need to be — I'll find the best bus, when to leave, and how crowded it'll be.",
    }
  ])

  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 150)
  }, [open])

  async function send(textArg) {
    const userText = toStr(textArg || input).trim()
    if (!userText) return

    const newMessages = [...messages, { role: 'user', content: userText }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    // History = everything except greeting and the message we just added
    const history = newMessages
      .slice(1, -1)
      .map(m => ({ role: m.role, content: toStr(m.content) }))

    try {
      const apiBase = import.meta.env.VITE_API_URL || ''
      const res = await fetch(`${apiBase}/api/travel-agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText, history }),
      })

      if (!res.ok) {
        // Give a clear message about what went wrong
        const statusText = res.status === 404
          ? "Travel agent endpoint not found (404). Make sure you replaced backend/app/main.py and restarted the server."
          : `Server error ${res.status}. Check the backend terminal for details.`
        setMessages(prev => [...prev, { role: 'assistant', content: statusText }])
        return
      }

      const data = await res.json()
      const reply = toStr(data.reply) || "I got an empty response. Please try again."
      setModelUsed(toStr(data.model_used))
      setMessages(prev => [...prev, { role: 'assistant', content: reply }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Connection error: ${e.message}. Make sure the backend is running at localhost:8000.`
      }])
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <>
      <button
        className={`ta-fab ${open ? 'ta-fab--open' : ''}`}
        onClick={() => setOpen(o => !o)}
        title="Transit agent"
        aria-label="Open transit assistant"
      >
        {open ? '✕' : '🗺️'}
      </button>

      {open && (
        <div className="ta-panel">
          <div className="ta-header">
            <span className="ta-header__icon">🗺️</span>
            <div>
              <div className="ta-header__title">Transit Agent</div>
              <div className="ta-header__sub">
                {modelUsed ? `Powered by ${modelUsed}` : 'IU Bloomington · Llama via HuggingFace'}
              </div>
            </div>
          </div>

          <div className="ta-messages">
            {messages.map((m, i) => <Message key={i} msg={m} />)}
            {loading && (
              <div className="ta-msg ta-msg--bot">
                <div className="ta-msg__avatar">🗺️</div>
                <div className="ta-msg__bubble ta-msg__bubble--typing">
                  <span /><span /><span />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {messages.length === 1 && (
            <div className="ta-suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="ta-suggestion" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          )}

          <div className="ta-input-row">
            <textarea
              ref={inputRef}
              className="ta-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Where are you going?"
              rows={1}
              disabled={loading}
            />
            <button
              className="ta-send"
              onClick={() => send()}
              disabled={!input.trim() || loading}
              aria-label="Send"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  )
}
