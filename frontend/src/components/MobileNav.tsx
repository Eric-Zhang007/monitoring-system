interface MobileNavProps {
  items: Array<{ id: string; label: string; icon: string }>;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export default function MobileNav({ items, activeTab, onTabChange }: MobileNavProps) {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-700
                  shadow-lg md:hidden z-40">
      <nav className="flex justify-around py-2">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            className={`
              flex flex-col items-center justify-center space-y-1 px-3 py-2 rounded-lg transition-all duration-200
              ${activeTab === item.id
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }
            `}
          >
            <span className="text-xl">{item.icon}</span>
            <span className="text-xs font-medium">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
