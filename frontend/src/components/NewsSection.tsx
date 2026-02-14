import { Newspaper, Clock, AlertCircle } from 'lucide-react';
import { NewsItem } from '../types';

const MOCK_NEWS: NewsItem[] = [
  {
    id: 1,
    title: '美联储暗示降息可能性增加，市场预期3月可能降息25基点',
    symbol: 'BTC',
    time: '2026-02-14T12:45:00Z',
    url: 'https://example.com/news/1',
    priority: 'high',
    sentiment: 'positive',
    is_important: true,
  },
  {
    id: 2,
    title: 'BTC 持续突破 $68,000，创下历史新高',
    symbol: 'BTC',
    time: '2026-02-14T12:30:00Z',
    url: 'https://example.com/news/2',
    priority: 'high',
    sentiment: 'positive',
    is_important: true,
  },
  {
    id: 3,
    title: '欧盟通过 MiCA 加密货币监管法案，行业迎来规范化发展',
    symbol: 'ETH',
    time: '2026-02-14T12:15:00Z',
    url: 'https://example.com/news/3',
    priority: 'high',
    sentiment: 'negative',
    is_important: true,
  },
  {
    id: 4,
    title: '苹果公司发布新款产品，供应链紧张问题仍未缓解',
    symbol: 'AAPL',
    time: '2026-02-14T12:00:00Z',
    url: 'https://example.com/news/4',
    priority: 'medium',
    sentiment: 'neutral',
    is_important: false,
  },
  {
    id: 5,
    title: '特斯拉股价创新高，电动化转型加速',
    symbol: 'TSLA',
    time: '2026-02-14T11:45:00Z',
    url: 'https://example.com/news/5',
    priority: 'medium',
    sentiment: 'positive',
    is_important: false,
  },
  {
    id: 6,
    title: '英伟达发布新一代 GPU，AI 算力再创新高',
    symbol: 'NVDA',
    time: '2026-02-14T11:30:00Z',
    url: 'https://example.com/news/6',
    priority: 'medium',
    sentiment: 'positive',
    is_important: false,
  },
  {
    id: 7,
    title: '谷歌发布最新 AI 模型，与 OpenAI 竞争加剧',
    symbol: 'GOOGL',
    time: '2026-02-14T11:15:00Z',
    url: 'https://example.com/news/7',
    priority: 'low',
    sentiment: 'neutral',
    is_important: false,
  },
];

export default function NewsSection() {
  const getTimeAgo = (time: string) => {
    const now = new Date();
    const newsTime = new Date(time);
    const diff = Math.floor((now.getTime() - newsTime.getTime()) / 1000 / 60);

    if (diff < 60) return `${diff} 分钟前`;
    if (diff < 1440) return `${Math.floor(diff / 60)} 小时前`;
    return `${Math.floor(diff / 1440)} 天前`;
  };

  const getPriorityClass = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300';
      default:
        return 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300';
    }
  };

  const getSentimentClass = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'border-l-4 border-green-500';
      case 'negative':
        return 'border-l-4 border-red-500';
      default:
        return 'border-l-4 border-slate-300 dark:border-slate-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Newspaper className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">新闻资讯</h2>
      </div>

      {/* Important News Highlight */}
      {MOCK_NEWS.filter(item => item.is_important).slice(0, 2).map(item => (
        <div
          key={item.id}
          className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20
                   rounded-xl p-6 shadow-md border border-blue-200 dark:border-blue-800"
        >
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <AlertCircle className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                {item.title}
              </h3>
              <div className="flex items-center space-x-3 text-sm text-slate-600 dark:text-slate-400">
                <span className="inline-flex items-center space-x-1">
                  <Clock className="w-4 h-4" />
                  <span>{getTimeAgo(item.time)}</span>
                </span>
                {item.symbol && (
                  <span className="bg-slate-200 dark:bg-slate-700 px-2 py-0.5 rounded text-xs font-medium">
                    {item.symbol}
                  </span>
                )}
              </div>
            </div>
            <span className={getPriorityClass(item.priority) + ' px-3 py-1 rounded-full text-xs font-semibold'}>
              重要
            </span>
          </div>
        </div>
      ))}

      {/* News List */}
      <div className="space-y-3">
        {MOCK_NEWS.map(item => (
          <article
            key={item.id}
            className={`
              bg-white dark:bg-slate-800 rounded-lg shadow-sm p-5 border border-slate-200 dark:border-slate-700
              hover:shadow-md transition-shadow duration-200 cursor-pointer
              ${getSentimentClass(item.sentiment)}
            `}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h3 className="text-base font-medium text-slate-900 dark:text-white mb-2
                                    hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                  {item.title}
                </h3>
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                  <span className="flex items-center space-x-1">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{getTimeAgo(item.time)}</span>
                  </span>
                  {item.symbol && (
                    <span className="bg-slate-100 dark:bg-slate-700 px-2 py-0.5 rounded">
                      {item.symbol}
                    </span>
                  )}
                  <span className={getPriorityClass(item.priority) + ' px-2 py-0.5 rounded'}>
                    {item.priority}
                  </span>
                </div>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
