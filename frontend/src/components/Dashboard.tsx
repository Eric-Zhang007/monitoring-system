import { useState, useEffect } from 'react';
import { ArrowUp, ArrowDown, Activity, TrendingUp, Wallet } from 'lucide-react';
import { useColorScheme } from '../contexts/ColorSchemeContext';

interface DashboardProps {
  activeSymbol: string;
  onSymbolChange: (symbol: string) => void;
}

const MOCK_PRICE_DATA = {
  BTC: { price: 67890.50, change: 2.35, volume: 28.5 },
  ETH: { price: 3456.78, change: 1.89, volume: 15.3 },
  AAPL: { price: 182.52, change: -0.45, volume: 52.1 },
  TSLA: { price: 248.67, change: 3.21, volume: 33.8 },
  NVDA: { price: 789.23, change: 4.56, volume: 21.9 },
  GOOGL: { price: 141.23, change: 0.78, volume: 25.4 },
};

export default function Dashboard({ activeSymbol, onSymbolChange }: DashboardProps) {
  const { scheme } = useColorScheme();
  const [currentPrice, setCurrentPrice] = useState(MOCK_PRICE_DATA[activeSymbol as keyof typeof MOCK_PRICE_DATA]);

  // Simulate price updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPrice(prev => ({
        ...prev,
        price: prev.price + (Math.random() - 0.5) * prev.price * 0.001,
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, [activeSymbol]);

  const getColorByChange = (change: number) => {
    if (scheme === 'cn') {
      return change >= 0
        ? 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20'
        : 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20';
    } else {
      return change >= 0
        ? 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20'
        : 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
    }
  };

  const colorClass = getColorByChange(currentPrice.change);

  return (
    <div className="space-y-6">
      {/* Symbol Selection */}
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-4 border border-slate-200 dark:border-slate-700">
        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          选择标的
        </label>
        <select
          value={activeSymbol}
          onChange={(e) => {
            onSymbolChange(e.target.value);
            setCurrentPrice(MOCK_PRICE_DATA[e.target.value as keyof typeof MOCK_PRICE_DATA]);
          }}
          className="w-full px-4 py-2 rounded-md border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {Object.keys(MOCK_PRICE_DATA).map(symbol => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </select>
      </div>

      {/* Main Price Card */}
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-sm text-slate-500 dark:text-slate-400 font-medium mb-1">当前价格</h2>
            <div className="flex items-baseline space-x-3">
              <span className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white">
                ${currentPrice.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
          </div>
          <div className={`
            px-3 py-1.5 rounded-full flex items-center space-x-1 ${colorClass}
          `}>
            {currentPrice.change >= 0 ? (
              <ArrowUp className="w-4 h-4" />
            ) : (
              <ArrowDown className="w-4 h-4" />
            )}
            <span className="text-sm font-semibold">
              {currentPrice.change >= 0 ? '+' : ''}{currentPrice.change.toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Additional Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">24H 成交量</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-white">
              {currentPrice.volume}亿
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">预测准确率</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-white">
              78.5%
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">情绪指数</p>
            <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
              {Math.floor(60 + Math.random() * 30)}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">更新时间</p>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              实时
            </p>
          </div>
        </div>
      </div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700
                         hover:shadow-lg transition-shadow duration-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">活跃度</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">高</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700
                         hover:shadow-lg transition-shadow duration-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">趋势</p>
              <p className="text-lg font-semibold text-green-600 dark:text-green-400">上涨</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700
                         hover:shadow-lg transition-shadow duration-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">建议</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">持有</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
