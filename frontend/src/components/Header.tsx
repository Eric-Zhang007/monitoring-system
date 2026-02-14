import { Menu } from 'lucide-react';
import { useColorScheme } from '../contexts/ColorSchemeContext';

interface HeaderProps {
  scheme: 'cn' | 'us';
  onMenuToggle: () => void;
}

export default function Header({ scheme, onMenuToggle }: HeaderProps) {
  const { toggleScheme } = useColorScheme();

  return (
    <header className="sticky top-0 z-50 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800 dark:text-white hidden sm:block">
                全网信息监测系统
              </h1>
              <h1 className="text-lg font-bold text-slate-800 dark:text-white sm:hidden">
                监测系统
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400 hidden md:block">
                Global Information Monitoring System
              </p>
            </div>
          </div>

          {/* Right Section - Color Toggle & Menu */}
          <div className="flex items-center space-x-3">
            {/* Color Scheme Toggle */}
            <button
              onClick={toggleScheme}
              className={`
                px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200
                ${scheme === 'cn'
                  ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                  : 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                }
              `}
              title={scheme === 'cn' ? '中国: 红涨绿跌' : '美国: 绿涨红跌'}
            >
              {scheme === 'cn' ? '🇨🇳 CN' : '🇺🇸 US'}
            </button>

            {/* Mobile Menu Button */}
            <button
              onClick={onMenuToggle}
              className="lg:hidden p-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              aria-label="Toggle menu"
            >
              <Menu className="w-6 h-6 text-slate-600 dark:text-slate-300" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
