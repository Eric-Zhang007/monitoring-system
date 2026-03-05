import { useEffect, useRef, useState } from 'react'

export function useSSE<T = any>(url: string | null) {
  const [lastEvent, setLastEvent] = useState<T | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!url) return
    const es = new EventSource(url)
    esRef.current = es
    es.onopen = () => {
      setConnected(true)
      setError(null)
    }
    es.onerror = () => {
      setConnected(false)
      setError('sse_disconnected')
    }
    es.onmessage = (evt) => {
      try {
        setLastEvent(JSON.parse(evt.data))
      } catch {
        setLastEvent((evt.data as any) || null)
      }
    }
    return () => {
      es.close()
      esRef.current = null
      setConnected(false)
    }
  }, [url])

  return { lastEvent, connected, error }
}
