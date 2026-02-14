import { TrendingUp, ShieldCheck, Zap } from 'lucide-react';

export default function Hero() {
  return (
    <section className="py-8 md:py-12 px-4">
      <div className="container mx-auto">
        <div className="text-center max-w-4xl mx-auto space-y-6">
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-slate-900 dark:text-white">
            实时监测
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600"> 全球金融</span>
            信息
          </h2>
          <p className="text-lg md:text-xl text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            利用 AI 技术实时分析新闻、价格和市场情绪，为您提供精准的价格预测和投资决策支持
          </p>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 md:6 mt-8">
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-md border border-slate-200 dark:border-slate-700">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center mx-auto mb-3">
                <TrendingUp className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white mb-1">实时预测</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">基于 AI 的价格趋势预测</p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-md border border-slate-200 dark:border-slate-700">
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center mx-auto mb-3">
                <ShieldCheck className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white mb-1">智能监控</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">24/7 全球新闻监测与情绪分析</p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-md border border-slate-200 dark:border-slate-700">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center mx-auto mb-3">
                <Zap className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white mb-1">NIM 加速</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">GPU 加速特征提取与推理</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
