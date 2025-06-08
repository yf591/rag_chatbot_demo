import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ベクトルストアの内容を確認
def check_vector_store():
    try:
        # 埋め込みモデルを初期化
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # ベクトルストアを読み込み
        vector_store = FAISS.load_local(
            "./vector_store", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # ドキュメント情報を読み込み
        with open("./vector_store/documents.pkl", "rb") as f:
            documents = pickle.load(f)
        
        print(f"Total documents: {len(documents)}")
        print("\nFirst 3 documents:")
        for i, doc in enumerate(documents[:3]):
            print(f"\nDocument {i+1}:")
            print(f"Source: {doc['metadata']['source']}")
            print(f"Content preview: {doc['content'][:200]}...")
        
        # 検索テスト
        print("\n" + "="*50)
        print("Search test:")
        
        test_queries = ["観光", "人口", "将来", "ビジョン"]
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = vector_store.similarity_search(query, k=2)
            for j, result in enumerate(results):
                print(f"  Result {j+1}: {result.page_content[:100]}...")
                print(f"  Source: {result.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_vector_store()

