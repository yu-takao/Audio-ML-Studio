import type { ReactNode } from 'react';
import { Waves, User, LogOut } from 'lucide-react';

interface NavItem {
  id: string;
  label: string;
  icon: ReactNode;
}

interface UserInfo {
  userId: string;
  username?: string;
}

interface SidebarProps {
  items: NavItem[];
  activeItem: string;
  onItemChange: (itemId: string) => void;
  user?: UserInfo;
  onSignOut?: () => void;
}

export function Sidebar({ items, activeItem, onItemChange, user, onSignOut }: SidebarProps) {
  return (
    <aside className="w-64 bg-zinc-900/80 border-r border-zinc-800 flex flex-col h-screen sticky top-0">
      {/* ロゴ */}
      <div className="p-5 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-lg font-bold text-white">Audio ML Studio</h1>
        </div>
      </div>

      {/* ナビゲーション */}
      <nav className="flex-1 p-4 space-y-2">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onItemChange(item.id)}
            className={`
              w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all text-left
              ${
                activeItem === item.id
                  ? 'bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 text-white border border-violet-500/30'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
              }
            `}
          >
            <span className={activeItem === item.id ? 'text-violet-400' : ''}>
              {item.icon}
            </span>
            {item.label}
          </button>
        ))}
      </nav>

      {/* ユーザー情報とサインアウト */}
      <div className="p-4 border-t border-zinc-800">
        {user ? (
          <div className="space-y-3">
            {/* ユーザー情報 */}
            <div className="flex items-center gap-3 px-3 py-2 bg-zinc-800/50 rounded-lg">
              <div className="p-1.5 rounded-full bg-violet-500/20">
                <User className="w-4 h-4 text-violet-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {user.username || 'ユーザー'}
                </p>
                <p className="text-xs text-zinc-500 truncate">
                  {user.userId.slice(0, 8)}...
                </p>
              </div>
            </div>
            
            {/* サインアウトボタン */}
            {onSignOut && (
              <button
                onClick={onSignOut}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                         text-sm font-medium text-zinc-400 hover:text-white
                         bg-zinc-800/30 hover:bg-zinc-800 border border-zinc-700/50
                         transition-all duration-200"
              >
                <LogOut className="w-4 h-4" />
                サインアウト
              </button>
            )}
          </div>
        ) : (
          <div className="text-xs text-zinc-600 text-center">
            Tire Sound Analysis
          </div>
        )}
      </div>
    </aside>
  );
}

