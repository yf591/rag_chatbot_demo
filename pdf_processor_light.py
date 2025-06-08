import fitz  # PyMuPDF
import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

class PDFProcessor:
    def __init__(self, pdf_directory: str = "./"):
        self.pdf_directory = pdf_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # チャンクサイズを小さく
            chunk_overlap=100,
            length_function=len,
        )
        # より軽量な埋め込みモデルを使用
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDFからテキストを抽出"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            # 最初の10ページのみ処理（デモ用）
            max_pages = min(10, len(doc))
            for page_num in range(max_pages):
                page = doc[page_num]
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_all_pdfs(self) -> List[Dict]:
        """すべてのPDFファイルを処理してテキストを抽出"""
        documents = []
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            print(f"Processing: {pdf_file}")
            
            text = self.extract_text_from_pdf(pdf_path)
            if text.strip():
                # テキストをチャンクに分割
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': pdf_file,
                        'chunk_id': i,
                        'metadata': {
                            'source': pdf_file,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    })
        
        print(f"Total documents processed: {len(documents)}")
        return documents
    
    def create_vector_store(self, documents: List[Dict], save_path: str = "./vector_store"):
        """ベクトルストアを作成"""
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        print("Creating vector store...")
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # ベクトルストアを保存
        vector_store.save_local(save_path)
        
        # ドキュメント情報も保存
        with open(f"{save_path}/documents.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        print(f"Vector store saved to: {save_path}")
        return vector_store

def main():
    # PDFプロセッサーを初期化
    processor = PDFProcessor()
    
    # PDFファイルを処理
    documents = processor.process_all_pdfs()
    
    if documents:
        # ベクトルストアを作成
        vector_store = processor.create_vector_store(documents)
        print("PDF processing completed successfully!")
    else:
        print("No documents found or processed.")

if __name__ == "__main__":
    main()

