import { Brain, TrendingUp, TrendingDown } from 'lucide-react';

const MOCK_PREDICTIONS = {
  BTC: [
    { horizon: '1h', direction: 'up', change: '+1.2%', confidence: 0.85, accuracy: 78 },
    { horizon: '1d', direction: 'up', change: '+3.5%', confidence: 0.73, accuracy: 73 },
    { horizon: '7d', direction: 'down', change: '-2.1%', confidence: 0.61, accuracy: 61 },
  ],
  ETH: [
    { horizon: '1h', direction: 'up', change: '+0.8%', confidence: 0.79, accuracy: 75 },
    { horizon: '1d', direction: 'neutral', change: '+0.2%', confidence: 0.65, accuracy: 68 },
    { horizon: '7d', direction: 'down', change: '-1.5%', confidence: 0.58, accuracy: 62 },
  ],
  AAPL: [
    { horizon: '1h', direction: 'down', change: '-0.5%', confidence: 0.71, accuracy: 70 },
    { horizon: '1d', direction: 'neutral', change: '+0.3%', confidence: 0.63, accuracy: 65 },
    { horizon: '7d', direction: 'up', change: '+2.8%', confidence: 0.67, accuracy: 67 },
  ],
  TSLA: [
    { horizon: '1h', direction: 'up', change: '+0.6%', confidence: 0.82, accuracy: 76 },
    { horizon: '1d', direction: 'up', change: '+2.1%', confidence: 0.74, accuracy: 71 },
    { horizon: '7d', direction: 'up', change: '+4.2%', confidence: 0.69, accuracy: 69 },
  ],
  NVDA: [
    { horizon: '1h', direction: 'up', change: '+1.5%', confidence: 0.87, accuracy: 80 },
    { horizon: '1d', direction: 'up', change: '+3.2%', confidence: 0.76, accuracy: 74 },
    { horizon: '7d', direction: 'up', change: '+6.8%', confidence: 0.72, accuracy: 72 },
  ],
  GOOGL: [
    { horizon: '1h', direction: 'neutral', change: '+0.1%', confidence: 0.62, accuracy: 64 },
    { horizon: '1d', direction: 'neutral', change: '+0.4%', confidence: 0.58, accuracy: 61 },
    { horizon: '7d', direction: 'down', change: '-1.2%', confidence: 0.55, accuracy: 59 },
  ],
};

interface PredictionsProps {
  symbol?: string;
}

export default function PredictionsSection({ symbol = 'BTC' }: PredictionsProps) {
  const predictions = MOCK_PREDICTIONS[symbol as keyof typeof MOCK_PREDICTIONS] || MOCK_PREDICTIONS.BTC;

  const getDirectionClass = (direction: string, scheme: 'cn' | 'us' = 'cn') => {
    if (direction === 'up') {
      return scheme === 'cn'
        ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
        : 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300';
    } else if (direction === 'down') {
      return scheme === 'cn'
        ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
        : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300';
    } else {
      return 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300';
    }
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return { text: '高', color: 'bg-green-500' };
    if (confidence >= 0.65) return { text: '中', color: 'bg-yellow-500' };
    return { text: '低', color: 'bg-red-500' };
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400" />
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">AI 预测分析 <span className="text-sm font-normal text-slate-500 dark:text-slate-400">| {symbol}</span></h2>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          未来走势预测
        </h3>

        {/* Predictions List */}
        <div className="space-y-4">
          {predictions.map((pred, index) => {
            const confidenceLevel = getConfidenceLevel(pred.confidence);
            return (
              <div
                key={index}
                className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 border border-slate-200 dark:border-slate-600"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {/* Horizon */}
                    <div className="w-16 text-center">
                      <div className="text-2xl font-bold text-slate-900 dark:text-white">
                        {pred.horizon}
                      </div>
                    </div>

                    {/* Direction & Change */}
                    <div className={`
                      px-3 py-1.5 rounded-lg flex items-center space-x-2
                      ${getDirectionClass(pred.direction, 'cn')}
                    `}>
                      {pred.direction === 'up' && <TrendingUp className="w-4 h-4" />}
                      {pred.direction === 'down' && <TrendingDown className="w-4 h-4" />}
                      {pred.direction === 'neutral' && <span className="font-semibold">•</span>}
                      <span className="font-semibold">{pred.change}</span>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="flex-1 max-w-xs mx-4">
                    <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                      <span>置信度</span>
                      <span>{(pred.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${confidenceLevel.color} transition-all duration-300`}
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Accuracy */}
                  <div className="text-center min-w-16">
                    <div className="text-sm font-medium text-green-600 dark:text-green-400">
                      {pred.accuracy}%
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      准确率
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Disclaimer */}
        <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
          <p className="text-sm text-amber-800 dark:text-amber-300">
            ⚠️ 免责声明：本预测仅供参考，不构成投资建议。投资有风险，需谨慎。
          </p>
        </div>
      </div>

      {/* Feature Integration Note */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20
                       rounded-xl p-6 border border-purple-200 dark:border-purple-800">
        <div className="flex items-start space-x-3">
          <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-base font-semibold text-slate-900 dark:text-white mb-2">
              NIM 特征提取
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              使用 NVIDIA NIM API 进行新闻文本特征提取，结合 LSTM 模型进行价格预测。
              特征向量维度：128 | 模型：LSTM (2层, 256 hidden)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
