import { useState, useEffect } from 'react';
import { Server, Database, Cpu, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import { SystemStatus } from '../types';

const MOCK_STATUS: SystemStatus = {
  system: {
    available: true,
    total_gpus: 2,
    gpu_memory: 48,
    utilization: 75,
  },
  services: {
    backend: 'running',
    redis_consumer: 'running',
    inference: 'running',
    training: 'running',
    collector: 'running',
  },
};

export default function MonitorPanel() {
  const [systemStatus] = useState<SystemStatus>(MOCK_STATUS);
  const [logs, setLogs] = useState<Array<{ id: number; level: string; message: string; timestamp: string }>>([]);

  // Simulate log updates
  useEffect(() => {
    const sampleLogs = [
      { level: 'info', message: '✅ Backend service started on port 8000' },
      { level: 'info', message: '✅ Redis connection established' },
      { level: 'info', message: '🎮 GPU 0 (Inference) initialized: 24GB' },
      { level: 'info', message: '🎮 GPU 1 (Training) initialized: 24GB' },
      { level: 'success', message: '✅ Model loaded on GPU 0 (Inference)' },
      { level: 'info', message: '📰 Starting news collector...' },
      { level: 'success', message: '✅ NIM feature extraction service ready' },
      { level: 'info', message: '🎓 Model training service started' },
      { level: 'info', message: '📊 Processing BTC price updates...' },
      { level: 'success', message: '✅ Prediction generated for BTC (confidence: 0.85)' },
    ];

    const generateTimestamp = () => {
      const now = new Date();
      return now.toISOString().replace('T', ' ').substring(0, 19);
    };

    const generateLogs = () => {
      let id = 1;
      const allLogs: Array<{ id: number; level: string; message: string; timestamp: string }> = [];

      for (let i = 0; i < sampleLogs.length; i++) {
        allLogs.push({
          id: id++,
          level: sampleLogs[i].level,
          message: sampleLogs[i].message,
          timestamp: generateTimestamp(),
        });
      }

      setLogs(allLogs);
    };

    generateLogs();

    // Simulate new logs
    const interval = setInterval(() => {
      const newLogs = [
        { level: 'info', message: '📊 Processing price updates...' },
        { level: 'success', message: `✅ Prediction generated (confidence: ${(0.6 + Math.random() * 0.3).toFixed(2)})` },
        { level: 'info', message: '📰 Processing news items...' },
        { level: 'info', message: '🎓 Training model epoch in progress...' },
      ];

      const randomLog = newLogs[Math.floor(Math.random() * newLogs.length)];
      setLogs(prev => [
        {
          id: Date.now(),
          level: randomLog.level,
          message: randomLog.message,
          timestamp: generateTimestamp(),
        },
        ...prev,
      ].slice(0, 50));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getLogIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      default:
        return <Activity className="w-4 h-4 text-blue-500" />;
    }
  };

  const getLogColor = (level: string) => {
    switch (level) {
      case 'error':
        return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
      case 'warning':
        return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20';
      case 'success':
        return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20';
      default:
        return 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Server className="w-6 h-6 text-orange-600 dark:text-orange-400" />
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">系统监控</h2>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* GPU Status */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <Cpu className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">GPU 状态</h3>
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
            {systemStatus.system.total_gpus} / {systemStatus.system.total_gpus}
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">在线</p>
          <div className="mt-3 space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-600 dark:text-slate-400">GPU 0 (推理)</span>
              <span className="text-green-600 dark:text-green-400">24GB</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-600 dark:text-slate-400">GPU 1 (训练)</span>
              <span className="text-green-600 dark:text-green-400">24GB</span>
            </div>
          </div>
        </div>

        {/* System Utilization */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">系统利用率</h3>
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
            {systemStatus.system.utilization}%
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">平均</p>
          <div className="mt-3 h-2 bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 dark:bg-blue-400 transition-all duration-300"
              style={{ width: `${systemStatus.system.utilization}%` }}
            />
          </div>
        </div>

        {/* Services Status */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <Server className="w-5 h-5 text-green-600 dark:text-green-400" />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">服务状态</h3>
          </div>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400 mb-1">
            正常
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {Object.values(systemStatus.services).filter(s => s === 'running').length} 运行中
          </p>
          <div className="mt-3 space-y-1">
            {Object.entries(systemStatus.services).slice(0, 2).map(([name, status]) => (
              <div key={name} className="flex items-center space-x-2 text-xs">
                <span className={`w-2 h-2 rounded-full ${status === 'running' ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-slate-600 dark:text-slate-400">{name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Memory Usage */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <Database className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">GPU 内存</h3>
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
            {systemStatus.system.gpu_memory}GB
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">总计</p>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded px-2 py-1 text-center">
              <div className="text-xs text-slate-500 dark:text-slate-400">GPU 0</div>
              <div className="text-sm font-semibold text-purple-600 dark:text-purple-400">24GB</div>
            </div>
            <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded px-2 py-1 text-center">
              <div className="text-xs text-slate-500 dark:text-slate-400">GPU 1</div>
              <div className="text-sm font-semibold text-indigo-600 dark:text-indigo-400">24GB</div>
            </div>
          </div>
        </div>
      </div>

      {/* Service Status Grid */}
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">服务详情</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {systemStatus.services && Object.entries(systemStatus.services).map(([name, status]) => (
            <div
              key={name}
              className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 border border-slate-200 dark:border-slate-600"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300 capitalize">
                  {name.replace('_', ' ')}
                </span>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                  status === 'running'
                    ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                    : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                }`}>
                  {status}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                {status === 'running' ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-red-500" />
                )}
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {status === 'running' ? '正常运行' : '已停止'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Log Viewer */}
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">系统日志</h3>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            最新 50 条
          </span>
        </div>
        <div className="bg-slate-950 dark:bg-slate-900 rounded-lg p-4 h-80 overflow-y-auto font-mono text-xs
                        scroll-smooth">
          {logs.map(log => (
            <div key={log.id} className={`flex items-start space-x-2 mb-1.5 ${getLogColor(log.level)}`}>
              <span className="flex-shrink-0 mt-0.5">{getLogIcon(log.level)}</span>
              <span className="text-slate-400 dark:text-slate-500 select-none">{log.timestamp}</span>
              <span className="flex-1 break-all">{log.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
