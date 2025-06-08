import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def fix_vector_store():
    """ベクトルストアの問題を修正"""
    try:
        # 埋め込みモデルを初期化
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # ドキュメント情報を読み込み
        with open("./vector_store/documents.pkl", "rb") as f:
            documents = pickle.load(f)
        
        print(f"Total documents: {len(documents)}")
        
        # テキストとメタデータを抽出
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        print("Creating new vector store...")
        # 新しいベクトルストアを作成
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # 保存
        vector_store.save_local("./vector_store_fixed")
        
        # ドキュメント情報も保存
        with open("./vector_store_fixed/documents.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        print("Fixed vector store saved!")
        
        # テスト
        print("\nTesting search...")
        test_queries = ["福山", "ビジョン", "将来", "観光"]
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = vector_store.similarity_search(query, k=2)
            for j, result in enumerate(results):
                print(f"  Result {j+1}: {result.page_content[:100]}...")
                print(f"  Source: {result.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_vector_store()

