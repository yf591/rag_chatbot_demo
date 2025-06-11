import streamlit as st
import os
import torch
from rag_system import RAGSystem

# Streamlitページ設定
st.set_page_config(
    page_title="福山市RAGチャットボット",
    page_icon="🏛️",
    layout="wide"
)

# セッション状態の初期化
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_system():
    """RAGシステムを初期化"""
    if st.session_state.rag_system is None:
        with st.spinner("RAGシステムを初期化中...（初回は2-3分かかります）"):
            try:
                # Streamlit secretsからHuggingFaceトークンを取得
                hf_token = st.secrets.get("huggingface", {}).get("token", None)
                if not hf_token:
                    # 環境変数からも試す
                    hf_token = os.getenv("HUGGINGFACE_TOKEN")
                
                if not hf_token:
                    st.error("HuggingFaceトークンが設定されていません。")
                    return False
                
                st.session_state.rag_system = RAGSystem(hf_token=hf_token)
                st.success("RAGシステムが正常に初期化されました！")
                
                # GPU使用状況を表示
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    st.info(f"🚀 GPU使用中: {gpu_name}")
                else:
                    st.warning("⚠️ CPUモードで実行中")
                
            except Exception as e:
                st.error(f"RAGシステムの初期化に失敗しました: {e}")
                return False
    return True

def main():
    st.title("🏛️ 福山市RAGチャットボット")
    st.markdown("福山市の公式資料に基づいて質問にお答えします。")
    
    # GPU情報表示
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)} | メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        st.sidebar.success(f"🚀 {gpu_info}")
    else:
        st.sidebar.warning("⚠️ CPUモードで実行中")
    
    # サイドバー
    with st.sidebar:
        st.header("📋 システム情報")
        st.markdown("""
        **利用可能な資料:**
        - 第四次福山市総合計画
        - 福山みらい創造ビジョン
        - 福山市観光パンフレット
        
        **使用モデル:**
        - LLM: tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1
        - 埋め込み: all-MiniLM-L6-v2
        - 量子化: 4bit QLoRA
        """)
        
        if st.button("🔄 システム再初期化"):
            st.session_state.rag_system = None
            st.session_state.chat_history = []
            st.rerun()
        
        # メモリ使用量表示
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.metric("GPU メモリ使用量", f"{memory_used:.1f}GB / {memory_total:.1f}GB")
    
    # RAGシステム初期化
    if not initialize_rag_system():
        st.stop()
    
    # チャット履歴表示
    if st.session_state.chat_history:
        st.subheader("💬 チャット履歴")
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}..."):
                st.markdown(f"**質問:** {question}")
                st.markdown(f"**回答:** {answer}")
                if sources:
                    st.markdown(f"**参考資料:** {', '.join(sources)}")
    
    # 質問入力
    st.subheader("❓ 質問を入力してください")
    
    # 質問例
    st.markdown("**質問例:**")
    example_questions = [
        "福山市の人口はどのくらいですか？",
        "福山市の主要な観光スポットを教えてください",
        "福山市の将来ビジョンについて教えてください",
        "福山市の産業について教えてください"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        col = cols[i % 2]
        if col.button(example, key=f"example_{i}"):
            st.session_state.current_question = example
    
    # 質問入力フォーム
    with st.form("question_form"):
        question = st.text_area(
            "質問を入力してください:",
            value=st.session_state.get('current_question', ''),
            height=100,
            placeholder="例: 福山市の観光スポットを教えてください"
        )
        submitted = st.form_submit_button("💬 質問する")
        
        if submitted and question.strip():
            with st.spinner("LLMで回答を生成中..."):
                try:
                    # RAGシステムで回答生成
                    result = st.session_state.rag_system.answer_question(question)
                    
                    # 結果表示
                    st.subheader("🤖 回答")
                    st.markdown(result['answer'])
                    
                    if result['sources']:
                        st.subheader("📚 参考資料")
                        for source in result['sources']:
                            st.markdown(f"- {source}")
                    
                    # 詳細情報の表示
                    if result['retrieved_docs']:
                        with st.expander("📄 検索された文書の詳細"):
                            for i, doc in enumerate(result['retrieved_docs']):
                                st.markdown(f"**文書 {i+1}:** {doc['metadata']['source']}")
                                st.markdown(f"```\n{doc['content'][:500]}...\n```")
                    
                    # チャット履歴に追加
                    st.session_state.chat_history.append((
                        question,
                        result['answer'],
                        result['sources']
                    ))
                    
                    # 現在の質問をクリア
                    if 'current_question' in st.session_state:
                        del st.session_state.current_question
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    福山市RAGチャットボット - GPU版 - 公式資料に基づく情報提供システム
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
