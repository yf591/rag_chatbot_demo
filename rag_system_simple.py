import os
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

class RAGSystemSimple:
    def __init__(self, vector_store_path: str = "./vector_store"):
        self.vector_store_path = vector_store_path
        
        # 埋め込みモデルを初期化（ベクトルストアと同じモデルを使用）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # ベクトルストアを読み込み
        self.vector_store = None
        self.documents = None
        self.load_vector_store()
    
    def load_vector_store(self):
        """ベクトルストアを読み込み"""
        try:
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # ドキュメント情報を読み込み
            with open(f"{self.vector_store_path}/documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            
            print("Vector store loaded successfully!")
        except Exception as e:
            print(f"Error loading vector store: {e}")
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict]:
        """クエリに関連するドキュメントを検索"""
        if not self.vector_store:
            return []
        
        try:
            # 類似度検索
            docs = self.vector_store.similarity_search(query, k=k)
            
            retrieved_docs = []
            for doc in docs:
                retrieved_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_simple_response(self, query: str, context_docs: List[Dict]) -> str:
        """シンプルな回答を生成（LLMなし）"""
        if not context_docs:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。"
        
        # 関連する文書の内容を組み合わせて回答を作成
        response = f"「{query}」に関する情報をお探しですね。\n\n"
        response += "福山市の資料から以下の関連情報が見つかりました：\n\n"
        
        for i, doc in enumerate(context_docs, 1):
            source = doc['metadata']['source']
            content = doc['content'][:300]  # 最初の300文字
            response += f"【{source}より】\n{content}...\n\n"
        
        response += "より詳細な情報については、上記の参考資料をご確認ください。"
        return response
    
    def answer_question(self, question: str) -> Dict:
        """質問に対する回答を生成"""
        # 関連ドキュメントを検索
        retrieved_docs = self.retrieve_documents(question, k=3)
        
        if not retrieved_docs:
            return {
                'answer': "申し訳ございませんが、関連する情報が見つかりませんでした。",
                'sources': [],
                'retrieved_docs': []
            }
        
        # 回答を生成
        answer = self.generate_simple_response(question, retrieved_docs)
        
        # ソース情報を抽出
        sources = list(set([doc['metadata']['source'] for doc in retrieved_docs]))
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_docs': retrieved_docs
        }

# テスト用の関数
def test_rag_system():
    """RAGシステムのテスト"""
    rag = RAGSystemSimple()
    
    test_questions = [
        "福山市の人口はどのくらいですか？",
        "福山市の観光スポットを教えてください",
        "福山市の将来ビジョンについて教えてください"
    ]
    
    for question in test_questions:
        print(f"\n質問: {question}")
        result = rag.answer_question(question)
        print(f"回答: {result['answer']}")
        print(f"参考資料: {result['sources']}")
        print("-" * 50)

if __name__ == "__main__":
    test_rag_system()

