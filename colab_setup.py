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
    """Google Colab最適化された依存関係をインストール"""
    print("=== 依存関係インストール ===")
    
    # Google Colab特有のFAISS問題対策
    print("🔧 Google Colab特有のFAISS問題対策...")
    try:
        # faiss-gpuがインストールされている場合は削除
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "faiss-gpu", "-y"], 
                      capture_output=True)
        # faiss-cpuをインストール（Google Colabで動作）
        subprocess.run([sys.executable, "-m", "pip", "install", "faiss-cpu"], 
                      check=True, capture_output=True, text=True)
        print("✅ FAISS問題対策完了")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ FAISS設定警告: {e}")
    
    # 通常の依存関係インストール
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ 依存関係のインストール完了")
    except subprocess.CalledProcessError as e:
        print(f"❌ インストールエラー: {e}")
        print(f"出力: {e.stdout}")
        print(f"エラー: {e.stderr}")
    print()

def create_vector_store():
    """ベクトルストアの確認・作成"""
    print("=== ベクトルストア確認・作成 ===")
    
    vector_store_path = Path("vector_store")
    faiss_file = vector_store_path / "index.faiss"
    documents_file = vector_store_path / "documents.pkl"
    
    if faiss_file.exists() and documents_file.exists():
        print("✅ ベクトルストア: 既存データ使用")
        return True
    
    print("🔧 ベクトルストアを作成中...")
    try:
        # PDFファイルの確認
        pdf_files = list(Path(".").glob("**/*.pdf"))
        if not pdf_files:
            print("⚠️ PDFファイルが見つかりません")
            print("📁 sample_documents/フォルダにPDFファイルを配置してください")
            return False
        
        print(f"📄 PDFファイル発見: {len(pdf_files)}個")
        
        # pdf_processor_light.pyを使用してベクトルストア作成
        if Path("pdf_processor_light.py").exists():
            result = subprocess.run([sys.executable, "pdf_processor_light.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ベクトルストア作成完了")
                return True
            else:
                print(f"❌ ベクトルストア作成エラー: {result.stderr}")
                return False
        else:
            print("❌ pdf_processor_light.py が見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ ベクトルストア作成中にエラー: {e}")
        return False
    
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
    
    # ベクトルストア確認・作成
    vector_store_created = create_vector_store()
    if not vector_store_created:
        print("⚠️ ベクトルストア作成に失敗しました")
        print("📝 手動でPDFファイルを配置後、再実行してください")
    
    verify_model_access()
    
    print("=" * 50)
    if vector_store_created:
        print("✅ セットアップ完了!")
        print()
        print("次のステップ:")
        print("1. 'streamlit run app.py' を実行")
        print("2. ngrokでトンネルを作成")
        print("3. 公開URLにアクセス")
    else:
        print("⚠️ セットアップ部分完了（ベクトルストア要作成）")
        print()
        print("次のステップ:")
        print("1. sample_documents/フォルダにPDFファイルを配置")
        print("2. 'python pdf_processor_light.py' を実行")
        print("3. 'streamlit run app.py' を実行")

if __name__ == "__main__":
    main()
