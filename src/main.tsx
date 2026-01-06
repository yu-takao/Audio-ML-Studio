import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Amplify } from 'aws-amplify'
import './index.css'
import App from './App.tsx'

// Amplify設定を読み込み（存在する場合のみ）
async function initializeAmplify() {
  try {
    // amplify_outputs.json はAmplifyデプロイ時に自動生成される
    const outputs = await import('../amplify_outputs.json')
    Amplify.configure(outputs.default)
    console.log('Amplify configured successfully')
  } catch (error) {
    // ローカル開発時やAmplify未設定時はスキップ
    console.log('Amplify configuration not found - running in local mode')
  }
}

// Amplify初期化後にアプリをレンダリング
initializeAmplify().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
