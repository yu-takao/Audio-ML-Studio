import { useState } from 'react';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import { Sidebar } from './components/Sidebar';
import { DataAugmentation } from './components/DataAugmentation';
import { ModelTraining } from './components/ModelTraining';
import { DataOrganizer } from './components/DataOrganizer';
import {
  Sparkles,
  Brain,
  AlertCircle,
  FolderCog,
} from 'lucide-react';

type TabId = 'training' | 'augmentation' | 'organizer';

// Authenticatorのカスタムコンポーネント
const authComponents = {
  Header() {
    return (
      <div className="text-center py-6">
        <h1 className="text-2xl font-bold text-white mb-2">Audio ML Studio</h1>
        <p className="text-zinc-400">サインインしてください</p>
      </div>
    );
  },
};

// Authenticatorの日本語設定
const formFields = {
  signIn: {
    username: {
      placeholder: 'メールアドレス',
      label: 'メールアドレス',
    },
    password: {
      placeholder: 'パスワード',
      label: 'パスワード',
    },
  },
  signUp: {
    username: {
      placeholder: 'メールアドレス',
      label: 'メールアドレス',
    },
    password: {
      placeholder: 'パスワード',
      label: 'パスワード',
    },
    confirm_password: {
      placeholder: 'パスワード（確認）',
      label: 'パスワード（確認）',
    },
  },
};

function AppContent({ user, signOut }: { user: { userId: string; username?: string }; signOut: () => void }) {
  const [activeTab, setActiveTab] = useState<TabId>('training');

  // File System Access API のサポートチェック
  const isSupported = 'showDirectoryPicker' in window;

  if (!isSupported) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center p-8">
        <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-8 max-w-md text-center">
          <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h1 className="text-xl font-bold text-white mb-2">非対応ブラウザ</h1>
          <p className="text-zinc-400">
            このアプリケーションはFile System Access APIを使用します。
            Chrome または Edge の最新版をお使いください。
          </p>
        </div>
      </div>
    );
  }

  const navItems = [
    {
      id: 'training',
      label: 'モデル構築',
      icon: <Brain className="w-5 h-5" />,
    },
    {
      id: 'augmentation',
      label: 'データ拡張',
      icon: <Sparkles className="w-5 h-5" />,
    },
    {
      id: 'organizer',
      label: 'データ整理',
      icon: <FolderCog className="w-5 h-5" />,
    },
  ];

  const getPageInfo = () => {
    switch (activeTab) {
      case 'training':
        return { title: 'モデル構築', description: '2D-CNNモデルを構築・訓練します' };
      case 'augmentation':
        return { title: 'データ拡張', description: '音声データの拡張を行います' };
      case 'organizer':
        return { title: 'データ整理', description: 'キーワードでファイルを分離・削除します' };
      default:
        return { title: '', description: '' };
    }
  };

  const pageInfo = getPageInfo();

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-900 via-zinc-900 to-violet-950 flex">
      {/* サイドバー */}
      <Sidebar
        items={navItems}
        activeItem={activeTab}
        onItemChange={(itemId) => setActiveTab(itemId as TabId)}
        user={user}
        onSignOut={signOut}
      />

      {/* メインコンテンツ */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-8 py-8">
          {/* ページタイトル */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white">{pageInfo.title}</h2>
            <p className="text-zinc-400 mt-1">{pageInfo.description}</p>
          </div>

          {/* タブコンテンツ */}
          {activeTab === 'training' && <ModelTraining userId={user.userId} />}
          {activeTab === 'augmentation' && <DataAugmentation />}
          {activeTab === 'organizer' && <DataOrganizer />}
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-900 via-zinc-900 to-violet-950">
      <Authenticator
        components={authComponents}
        formFields={formFields}
        hideSignUp={false}
      >
        {({ signOut, user }) => (
          <AppContent 
            user={{ 
              userId: user?.userId || '', 
              username: user?.signInDetails?.loginId 
            }} 
            signOut={signOut || (() => {})} 
          />
        )}
      </Authenticator>
    </div>
  );
}

export default App;
