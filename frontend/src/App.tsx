import { useState } from 'react'
import { useEffect } from 'react'
import Header from './components/Header'
import Hero from './components/Hero'
import Dashboard from './components/Dashboard'
import NewsSection from './components/NewsSection'
import PredictionsSection from './components/PredictionsSection'
import MonitorPanel from './components/MonitorPanel'
import V2Lab from './components/V2Lab'
import MobileNav from './components/MobileNav'
import { useWebSocket } from './hooks/useWebSocket'
import { useColorScheme } from './contexts/ColorSchemeContext'
import { WebSocketMessage } from './types'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const { scheme } = useColorScheme()
  const [activeSymbol, setActiveSymbol] = useState<string>('BTC')
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  // WebSocket connection
  const { isConnected, subscribe } = useWebSocket()

  // Handle incoming WebSocket messages
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const message: WebSocketMessage = JSON.parse(event.data)
      console.log('Received message:', message)
    }

    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [])

  // Subscribe to active symbol
  useEffect(() => {
    if (isConnected && activeSymbol) {
      subscribe(activeSymbol)
    }
  }, [isConnected, activeSymbol, subscribe])

  const navigationItems = [
    { id: 'dashboard', label: '仪表盘', icon: '📊' },
    { id: 'predictions', label: '预测分析', icon: '📈' },
    { id: 'v2lab', label: 'V2实验台', icon: '🧪' },
    { id: 'news', label: '新闻资讯', icon: '📰' },
    { id: 'monitor', label: '系统监控', icon: '🔍' },
  ]

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard activeSymbol={activeSymbol} onSymbolChange={setActiveSymbol} />
      case 'predictions':
        return <PredictionsSection symbol={activeSymbol} />
      case 'v2lab':
        return <V2Lab />
      case 'news':
        return <NewsSection />
      case 'monitor':
        return <MonitorPanel />
      default:
        return <Dashboard activeSymbol={activeSymbol} onSymbolChange={setActiveSymbol} />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header - Desktop & Mobile */}
      <Header
        scheme={scheme}
        onMenuToggle={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
      />

      {/* Hero Section */}
      <Hero />

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6 md:py-8">
        {/* Desktop Navigation */}
        <div className="hidden mb-6">
          <nav className="flex space-x-2 bg-white dark:bg-slate-800 rounded-lg p-2 shadow-sm">
            {navigationItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`
                  flex items-center space-x-2 px-4 py-2 rounded-md transition-all duration-200
                  ${activeTab === item.id
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                  }
                `}
              >
                <span className="text-lg">{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Mobile Navigation */}
        <MobileNav
          items={navigationItems}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />

        {/* Content Area */}
        <div className="animate-in fade-in duration-300">
          {renderContent()}
        </div>
      </main>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 xs:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}
    </div>
  )
}

export default App
