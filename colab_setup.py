#!/usr/bin/env python3
"""
Google Colab環境でRAGチャットボットを実行するためのセットアップスクリプト

使用方法
1. Google Colab ProでGPU（T4/L4/A100）を有効化
2. 環境変数（HUGGINGFACE_TOKEN, NGROK_TOKEN）を設定
3. このスクリプトを実行
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def check_gpu():
    """GPU利用可能性をチェック"""
    print("=== GPU情報 ===")
    if torch.cuda.is_available():
        print(f"✅ CUDA利用可能")
        print(f"GPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"メモリ: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")
    else:
        print("❌ CUDA利用不可 - CPUモードで実行されます")
    print()

def check_environment():
    """環境変数をチェック"""
    print("=== 環境変数チェック ===")
    required_vars = ["HUGGINGFACE_TOKEN", "NGROK_TOKEN"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: 設定済み")
        else:
            print(f"❌ {var}: 未設定")
    print()

def check_files():
    """必要なファイルの存在確認"""
    print("=== ファイル確認 ===")
    required_files = [
        "app.py",
        "rag_system.py", 
        "vector_store/index.faiss",
        "vector_store/documents.pkl",
        ".env"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: 存在")
        else:
            print(f"❌ {file_path}: 不足")
    print()

def setup_streamlit_secrets():
    """Streamlit用のsecrets.tomlファイルを設定"""
    print("=== Streamlit設定 ===")
    
    # .streamlitディレクトリ作成
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # secrets.tomlファイル作成
    secrets_content = f"""[huggingface]
token = "{os.getenv('HUGGINGFACE_TOKEN', '')}"

[ngrok]
token = "{os.getenv('NGROK_TOKEN', '')}"
"""
    
    secrets_path = streamlit_dir / "secrets.toml"
    with open(secrets_path, "w") as f:
        f.write(secrets_content)
    
    print(f"✅ {secrets_path}: 作成完了")
    print()

def install_dependencies():
    """依存関係をインストール"""
    print("=== 依存関係インストール ===")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ 依存関係のインストール完了")
    except subprocess.CalledProcessError as e:
        print(f"❌ インストールエラー: {e}")
        print(f"出力: {e.stdout}")
        print(f"エラー: {e.stderr}")
    print()

def verify_model_access():
    """モデルアクセス確認"""
    print("=== モデルアクセス確認 ===")
    try:
        from huggingface_hub import login, whoami
        
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
            user_info = whoami()
            print(f"✅ HuggingFace認証成功: {user_info['name']}")
        else:
            print("❌ HuggingFaceトークンが設定されていません")
    except Exception as e:
        print(f"❌ 認証エラー: {e}")
    print()

def main():
    """メイン実行関数"""
    print("🚀 Google Colab RAGチャットボット セットアップ")
    print("=" * 50)
    
    check_gpu()
    check_environment()
    check_files()
    setup_streamlit_secrets()
    install_dependencies()
    verify_model_access()
    
    print("=" * 50)
    print("✅ セットアップ完了!")
    print()
    print("次のステップ:")
    print("1. 'streamlit run app.py' を実行")
    print("2. ngrokでトンネルを作成")
    print("3. 公開URLにアクセス")

if __name__ == "__main__":
    main()
