import os
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

class RAGSystem:
    def __init__(self, vector_store_path: str = "./vector_store", hf_token: Optional[str] = None):
        self.vector_store_path = vector_store_path
        self.hf_token = hf_token
        # デフォルトのLLMモデルをHugging Faceの公開モデルに変更
        self.model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1"
        
        # 埋め込みモデルを初期化（GPUが利用可能な場合はGPU使用）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        # ベクトルストアを読み込み
        self.vector_store = None
        self.documents = None
        self.load_vector_store()
        
        # LLMモデルとトークナイザー
        self.model = None
        self.tokenizer = None
        self.load_llm_model()
    
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
    
    def load_llm_model(self):
        """LLMモデルを読み込み"""
        try:
            print("Loading LLM model...")
            
            # GPU利用可能性をチェック
            if not torch.cuda.is_available():
                print("Warning: CUDA is not available. Consider using CPU-optimized model.")
            
            # QLoRA設定（GPU使用時）
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            
            # モデル読み込み
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # トークナイザー読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                token=self.hf_token
            )
            
            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("LLM model loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            # フォールバック: モデル読み込みに失敗した場合はダミー応答
            self.model = None
            self.tokenizer = None
    
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
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """LLMを使用して回答を生成"""
        if not self.model or not self.tokenizer:
            return "申し訳ございませんが、現在モデルが利用できません。"
        
        try:
            # コンテキストを構築
            context = ""
            for doc in context_docs:
                context += f"【{doc['metadata']['source']}】\n{doc['content']}\n\n"
            
            # プロンプトを構築
            prompt = f"""### 指示
以下の福山市に関する資料を参考にして、質問に答えてください。

### 参考資料
{context}

### 質問
{query}

### 回答：
"""
            
            # トークン化
            tokenized_input = self.tokenizer.encode(
                prompt, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成パラメータの最適化
            with torch.no_grad():
                outputs = self.model.generate(
                    tokenized_input,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(tokenized_input)
                )[0]
            
            # デコード
            response = self.tokenizer.decode(
                outputs[tokenized_input.size(1):], 
                skip_special_tokens=True
            )
            
            return response.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"回答生成中にエラーが発生しました: {str(e)}"
    
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
        answer = self.generate_response(question, retrieved_docs)
        
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
    rag = RAGSystem()
    
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

