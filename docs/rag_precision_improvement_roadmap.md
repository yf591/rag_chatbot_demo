# RAG精度改善ロードマップ

## 🎯 **プロジェクト概要**

本プロジェクト「rag-chatbot-demo」は、自治体向けRAG（Retrieval-Augmented Generation）チャットボットの**デモンストレーション・プロトタイプ**です。

### デモ版としての位置づけ
- **目的**: RAGシステムの概念実証と技術的可能性の提示
- **現状**: 基本的な機能実装とGoogle Colab対応を優先
- **将来**: プロダクション環境への発展可能性を考慮した改善案を提示

---

## 📊 **現在の実装状況分析**

### 軽量版実装 (`pdf_processor_light.py`)
```python
class PDFProcessor:
    def __init__(self, pdf_directory: str = "./"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 🔍 改善点1: 小さすぎるチャンクサイズ
            chunk_overlap=100,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 🔍 改善点2: 軽量モデル
            model_kwargs={'device': 'cpu'}
        )
```

### 標準版実装 (`pdf_processor.py`)
```python
class PDFProcessor:
    def __init__(self, pdf_directory: str = "./"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # ✅ 改善済み: より適切なサイズ
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",  # ✅ 改善済み: 多言語高性能モデル
            model_kwargs={'device': 'cpu'}
        )
```

### 現在の制限事項
1. **PDF処理**: 軽量版では10ページのみ処理（デモ用制限）
2. **検索精度**: シンプルなベクトル検索のみ
3. **生成品質**: 基本的なプロンプトのみ
4. **評価システム**: 定量的な精度測定なし

---

## 🚀 **段階別改善ロードマップ**

## Phase 1: 即座に実装可能な改善（CPU環境対応）

### 1.1 PDF処理の強化

#### **ファイル**: `pdf_processor_light.py` → `pdf_processor_enhanced.py`

このファイルは既存の軽量版PDF処理器を改良し、より高度なPDF処理機能を持つように修正を加えたものを想定しています。全ページ処理、テキスト前処理の強化、メタデータの充実により、検索精度の向上を目指します。

**現在のコード**

現在の実装では以下の課題があります。
- 最初の10ページのみ処理（デモ用制限）
- 基本的なテキスト抽出のみでメタデータが不足
- テキストの前処理が不十分でノイズが混入

```python
def extract_text_from_pdf(self, pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    # 最初の10ページのみ処理（デモ用）
    max_pages = min(10, len(doc))
    for page_num in range(max_pages):
        page = doc[page_num]        text += page.get_text()
```

**改善後のコード**

この改善により以下の効果が期待できます。
- 全ページ処理による情報漏れの防止
- ページごとのメタデータ追加による検索精度向上
- テキスト前処理によるノイズ除去と品質向上
- 福山市固有のノイズパターンに対応した正規化

```python
def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
    """PDFからページごとにテキストとメタデータを抽出"""
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):  # 全ページ処理
        page = doc[page_num]
        text = page.get_text()
        
        # テキスト前処理
        text = self._preprocess_text(text)
        
        pages_data.append({
            'text': text,
            'page_number': page_num + 1,
            'source_file': os.path.basename(pdf_path),
            'total_pages': len(doc)
        })
    
    doc.close()
    return pages_data

def _preprocess_text(self, text: str) -> str:
    """テキストの前処理と正規化"""
    import re
    
    # 不要な改行・空白の正規化
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 文書特有のノイズ除去
    text = re.sub(r'ページ\s*\d+', '', text)  # ページ番号除去
    text = re.sub(r'福山市\s*\d{4}年', '', text)  # 年度表記の統一
    
    return text.strip()
```

### 1.2 チャンク戦略の最適化

#### **ファイル**: `pdf_processor_enhanced.py`

このクラスでは、従来の単純なテキスト分割から、意味的なコンテキストを保持する高度なチャンク戦略を実装します。日本語特有の文書構造に配慮した分割と、チャンク品質の評価機能により、検索時により関連性の高い文書片を取得できるようになると考えています。

```python
class EnhancedPDFProcessor:
    def __init__(self, pdf_directory: str = "./"):
        # 意味的分割を考慮したスプリッター
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,           # 最適化されたサイズ
            chunk_overlap=150,        # 文脈保持のための重複
            length_function=len,
            separators=[              # 日本語に最適化された区切り文字
                "\n\n",              # 段落区切り
                "\n",                # 行区切り
                "。",                 # 句点
                "、",                 # 読点
                " ",                 # スペース
                ""                   # 文字単位
            ]
        )
        
    def create_semantic_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """意味的なチャンク分割"""
        chunks = self.text_splitter.split_text(text)
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # チャンクの品質評価
            if len(chunk.strip()) < 50:  # 短すぎるチャンクをスキップ
                continue
                
            enhanced_chunk = {
                'content': chunk,
                'metadata': {
                    **metadata,
                    'chunk_id': i,
                    'chunk_length': len(chunk),
                    'chunk_quality_score': self._calculate_chunk_quality(chunk)
                }
            }
            enhanced_chunks.append(enhanced_chunk)
            
        return enhanced_chunks
        
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """チャンクの品質スコア計算"""
        # 基本的な品質指標
        score = 0.0
        
        # 長さスコア（適切な長さに高得点）
        length_score = min(len(chunk) / 500, 1.0)
        
        # 意味性スコア（句読点の存在）
        semantic_score = (chunk.count('。') + chunk.count('、')) / max(len(chunk) / 100, 1)
        
        # 情報密度スコア（数字・固有名詞の存在）
        import re
        info_density = len(re.findall(r'[0-9一二三四五六七八九十百千万]+|福山|市', chunk)) / max(len(chunk) / 50, 1)
        
        score = (length_score + semantic_score + info_density) / 3
        return min(score, 1.0)
```

### 1.3 埋め込みモデルの段階的改善

#### **ファイル**: `embedding_manager.py` (新規作成)

このファイルを新規作成することで、以下の効果が期待できます。
- 埋め込みモデルの一元管理と性能特性の明確化
- 用途に応じた最適なモデル選択の自動化
- モデル変更時の影響範囲の局所化
- 将来的なモデル拡張への柔軟な対応

```python
class EmbeddingManager:
    """埋め込みモデルの管理と最適化"""
    
    MODELS = {
        'light': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'size': '80MB',
            'performance': 'Low',
            'language': 'Multilingual'
        },
        'standard': {
            'name': 'intfloat/multilingual-e5-large', 
            'size': '1.2GB',
            'performance': 'High',
            'language': 'Multilingual'
        },
        'japanese': {
            'name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'size': '420MB', 
            'performance': 'Medium-High',
            'language': 'Japanese-optimized'
        }
    }
    
    def __init__(self, model_type: str = 'standard', device: str = 'cpu'):
        self.model_config = self.MODELS[model_type]
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_config['name'],
            model_kwargs={'device': device}
        )
        
    def get_model_info(self) -> Dict:
        """現在のモデル情報を取得"""
        return {
            'model_name': self.model_config['name'],
            'estimated_size': self.model_config['size'],
            'expected_performance': self.model_config['performance'],
            'language_support': self.model_config['language']
        }
```

---

## Phase 2: 検索精度の向上

### 2.1 ハイブリッド検索の実装

#### **ファイル**: `hybrid_retriever.py` (新規作成)

このファイルを新規作成することで、以下の効果が期待できます。
- Dense検索とSparse検索の長所を組み合わせた高精度検索
- 語彙的マッチング（BM25）と意味的マッチング（ベクトル）の相補的活用
- 検索結果の多様性向上と検索漏れの削減
- 福山市固有の用語に対する重み付け機能

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class HybridRetriever:
    """Dense（ベクトル）+ Sparse（BM25）のハイブリッド検索"""
    
    def __init__(self, vector_store, documents: List[Dict]):
        # Dense retriever (現在のFAISS)
        self.dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}
        )
        
        # Sparse retriever (BM25)
        texts = [doc['content'] for doc in documents]
        self.sparse_retriever = BM25Retriever.from_texts(
            texts, k=10
        )
        
        # アンサンブル検索器
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.7, 0.3]  # Dense:Sparse = 7:3
        )
        
    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """ハイブリッド検索による文書取得"""
        # アンサンブル検索実行
        docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # 重複除去と再ランキング
        unique_docs = self._remove_duplicates(docs[:k*2])
        reranked_docs = self._rerank_documents(query, unique_docs)
        
        return reranked_docs[:k]
        
    def _rerank_documents(self, query: str, docs: List) -> List:
        """簡易リランキング"""
        scored_docs = []
        for doc in docs:
            # 単純なキーワードマッチングスコア
            content = doc.page_content.lower()
            query_lower = query.lower()
            
            # キーワード一致度
            keyword_score = sum(1 for word in query_lower.split() if word in content)
            
            # 福山市関連用語の重み付け
            fukuyama_terms = ['福山', '市', '自治体', '行政', '市民']
            fukuyama_score = sum(2 for term in fukuyama_terms if term in content)
            
            total_score = keyword_score + fukuyama_score
            scored_docs.append((doc, total_score))
            
        # スコア順でソート
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
```

### 2.2 検索クエリ拡張

#### **ファイル**: `query_enhancer.py` (新規作成)

このファイルを新規作成することで、以下の効果が期待できます。
- 福山市特有の用語・同義語による検索拡張
- ユーザーの曖昧な質問に対する検索精度向上
- 行政用語と市民用語の橋渡し機能
- コンテキスト依存の最適なクエリ変換

```python
class QueryEnhancer:
    """検索クエリの拡張と最適化"""
    
    def __init__(self):
        # 福山市特有の用語辞書
        self.fukuyama_synonyms = {
            '福山': ['福山市', 'ふくやま'],
            '観光': ['観光地', '名所', '観光スポット', '見どころ'],
            '計画': ['基本計画', '総合計画', 'プラン', '構想'],
            '市民': ['住民', '市民の皆様', '住民の方'],
            '行政': ['市役所', '自治体', '福山市役所']
        }
        
    def expand_query(self, query: str) -> str:
        """クエリを拡張して検索精度を向上"""
        expanded_terms = []
        
        for word in query.split():
            expanded_terms.append(word)
            
            # 同義語の追加
            if word in self.fukuyama_synonyms:
                expanded_terms.extend(self.fukuyama_synonyms[word])
        
        # 重複除去して結合
        unique_terms = list(set(expanded_terms))
        return ' '.join(unique_terms)
        
    def optimize_query_for_context(self, query: str) -> str:
        """コンテキストに応じたクエリ最適化"""
        # 福山市コンテキストの明示的追加
        if '福山' not in query and '市' not in query:
            query = f"福山市 {query}"
            
        return query
```

---

## Phase 3: 生成品質の向上

### 3.1 プロンプトエンジニアリング

#### **ファイル**: `rag_system_enhanced.py`

**現在のコード** (`rag_system_simple.py`)

現在の実装では以下の課題があります。
- 単一の汎用プロンプトのみで質問の性質を考慮していない
- 回答の信頼度や根拠が不明確
- 市民向けの分かりやすさが不十分

```python
def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
    context = "\n".join([doc['content'] for doc in retrieved_docs])
    
    prompt = f"""
以下の文書を参考に、質問に答えてください。

参考文書:
{context}

質問: {query}

回答:"""
```

**改善後のコード**

この改善により以下の効果が期待できます。
- 質問カテゴリ（観光・計画・一般）に応じた最適化されたプロンプト
- 回答の信頼度スコアと根拠資料の明示
- 市民にも分かりやすい丁寧な説明形式
- 資料名の明示による透明性向上

```python
class EnhancedRAGSystem:
    def __init__(self):
        self.prompt_templates = {
            'default': """あなたは福山市の行政情報に関する専門アシスタントです。

【参考資料】
{context}

【質問】{query}

【回答指針】
1. 提供された福山市の公式文書の内容のみに基づいて回答してください
2. 根拠となる資料名を明示してください  
3. 不明な点は「提供された資料では確認できません」と回答してください
4. 市民の方にも分かりやすい言葉で説明してください

【回答】:""",
            
            'tourism': """福山市の観光情報について、以下の資料に基づいてご案内いたします。

【観光資料】
{context}

【お問い合わせ】{query}

【観光案内】
福山市の魅力をお伝えするため、公式パンフレットの情報をもとに回答いたします：""",
            
            'planning': """福山市の将来計画・政策について説明いたします。

【計画資料】
{context}

【ご質問】{query}

【政策説明】
福山市の公式計画書に基づいて、以下のようにお答えします："""
        }
        
    def select_prompt_template(self, query: str) -> str:
        """質問内容に応じたプロンプトテンプレート選択"""
        if any(word in query for word in ['観光', '名所', '見どころ', 'スポット']):
            return self.prompt_templates['tourism']
        elif any(word in query for word in ['計画', '政策', '将来', 'ビジョン']):
            return self.prompt_templates['planning']
        else:
            return self.prompt_templates['default']
            
    def generate_response_with_confidence(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        """信頼度付きレスポンス生成"""
        # 検索文書の関連度評価
        relevance_scores = [doc.get('relevance_score', 0.5) for doc in retrieved_docs]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # コンテキスト構築
        context = self._build_enhanced_context(retrieved_docs)
        
        # プロンプト選択
        prompt_template = self.select_prompt_template(query)
        prompt = prompt_template.format(context=context, query=query)
        
        # レスポンス生成（実装は使用するLLMによる）
        response = self._generate_llm_response(prompt)
        
        return {
            'answer': response,
            'confidence_score': avg_relevance,
            'source_documents': [doc.get('source', 'Unknown') for doc in retrieved_docs],
            'context_quality': self._assess_context_quality(context)
        }
        
    def _build_enhanced_context(self, docs: List[Dict]) -> str:
        """強化されたコンテキスト構築"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            source = doc.get('source', 'Unknown')
            content = doc.get('content', '')
            
            context_part = f"【資料{i}】{source}\n{content}\n"
            context_parts.append(context_part)
            
        return "\n".join(context_parts)
```

---

## Phase 4: 評価・監視システム

### 4.1 精度評価システム

#### **ファイル**: `evaluation_system.py` (新規作成)

このファイルを新規作成することで、以下の効果が期待できます。
- RAGシステムの性能を定量的に測定・監視
- 改善施策の効果を客観的に評価
- カテゴリ別の性能分析による弱点の特定
- 継続的改善のためのデータドリブンな意思決定支援

```python
class RAGEvaluationSystem:
    """RAGシステムの精度評価"""
    
    def __init__(self):
        self.test_queries = [
            {
                'query': '福山市の観光スポットを教えて',
                'expected_sources': ['fukuyama_tourism_pamphlet_etto.pdf'],
                'category': 'tourism'
            },
            {
                'query': '福山市の将来計画について',
                'expected_sources': ['fukuyama_comprehensive_plan_basic_concept.pdf'],
                'category': 'planning'
            }
            # 追加のテストケース...
        ]
        
    def evaluate_retrieval_accuracy(self, rag_system) -> Dict:
        """検索精度の評価"""
        results = {
            'total_queries': len(self.test_queries),
            'correct_retrievals': 0,
            'precision_scores': [],
            'recall_scores': [],
            'category_performance': {}
        }
        
        for test_case in self.test_queries:
            retrieved_docs = rag_system.retrieve_documents(test_case['query'])
            retrieved_sources = [doc.get('source', '') for doc in retrieved_docs]
            
            # 精度計算
            precision = self._calculate_precision(
                retrieved_sources, 
                test_case['expected_sources']
            )
            recall = self._calculate_recall(
                retrieved_sources,
                test_case['expected_sources'] 
            )
            
            results['precision_scores'].append(precision)
            results['recall_scores'].append(recall)
            
            # カテゴリ別性能記録
            category = test_case['category']
            if category not in results['category_performance']:
                results['category_performance'][category] = []
            results['category_performance'][category].append({
                'precision': precision,
                'recall': recall
            })
            
        # 平均精度計算
        results['avg_precision'] = sum(results['precision_scores']) / len(results['precision_scores'])
        results['avg_recall'] = sum(results['recall_scores']) / len(results['recall_scores'])
        results['f1_score'] = 2 * (results['avg_precision'] * results['avg_recall']) / (results['avg_precision'] + results['avg_recall'])
        
        return results
        
    def generate_evaluation_report(self, results: Dict) -> str:
        """評価レポート生成"""
        report = f"""
# RAGシステム精度評価レポート

## 全体性能
- **平均精度 (Precision)**: {results['avg_precision']:.3f}
- **平均再現率 (Recall)**: {results['avg_recall']:.3f}  
- **F1スコア**: {results['f1_score']:.3f}

## カテゴリ別性能
"""
        for category, performances in results['category_performance'].items():
            avg_precision = sum(p['precision'] for p in performances) / len(performances)
            avg_recall = sum(p['recall'] for p in performances) / len(performances)
            
            report += f"""
### {category.title()}
- 精度: {avg_precision:.3f}
- 再現率: {avg_recall:.3f}
"""
        
        return report
```

---

## 🔧 **実装ガイドライン**

### ローカル環境での段階的導入

#### **Step 1**: 基本改善の適用
```bash
# 新しい処理器での実行
python pdf_processor_enhanced.py

# ベクトルストア再構築
python scripts/rebuild_vector_store.py
```

#### **Step 2**: アプリケーションの更新

既存のシンプルアプリを拡張し、改善されたRAGシステムを統合します。フォールバック機能により、従来システムとの互換性を保ちながら段階的な移行を可能にします。

```python
# app_simple.pyの更新例
from hybrid_retriever import HybridRetriever
from rag_system_enhanced import EnhancedRAGSystem

class ImprovedSimpleApp:
    def __init__(self):
        # 従来のシステムとの互換性保持
        self.basic_rag = RAGSystemSimple()  # フォールバック用
        self.enhanced_rag = EnhancedRAGSystem()  # 改善版
        
    def get_response(self, query: str, use_enhanced: bool = True):
        if use_enhanced:
            try:
                return self.enhanced_rag.generate_response_with_confidence(query)
            except Exception as e:
                print(f"Enhanced system error: {e}")
                # フォールバック
                return self.basic_rag.get_response(query)
        else:
            return self.basic_rag.get_response(query)
```

#### **Step 3**: 性能評価の実行

改善されたRAGシステムの性能を定量的に測定し、改善効果を検証します。この評価により、さらなる改善の方向性を決定できます。

```python
# 改善効果の測定
evaluator = RAGEvaluationSystem()
results = evaluator.evaluate_retrieval_accuracy(enhanced_rag_system)
report = evaluator.generate_evaluation_report(results)
print(report)
```

---

## 📈 **期待される改善効果**

### 定量的目標（仮）

| 指標 | 現在 (推定) | Phase 1後 | Phase 2後 | Phase 3後 |
|------|-------------|------------|------------|------------|
| 検索精度 (Precision) | 0.6 | 0.75 | 0.85 | 0.90 |
| 検索再現率 (Recall) | 0.5 | 0.65 | 0.80 | 0.85 |
| 応答品質 (主観) | 普通 | 良好 | 優秀 | 非常に優秀 |
| 処理時間 | 基準 | +20% | +40% | +60% |

### 定性的改善

- **Phase 1**: より多くの文書内容の活用、文脈保持の向上
- **Phase 2**: 関連性の高い文書の発見、検索漏れの削減  
- **Phase 3**: 自然で的確な回答、信頼性の向上
- **Phase 4**: 継続的改善のための評価基盤

---

## 🛠️ **開発推奨スケジュール（仮）**

### 短期 (1-2週間)
- [ ] `pdf_processor_enhanced.py` の実装
- [ ] チャンク戦略の最適化
- [ ] 基本的なプロンプト改善

### 中期 (1-2ヶ月)  
- [ ] ハイブリッド検索の導入
- [ ] クエリ拡張機能の実装
- [ ] 評価システムの構築

### 長期 (3-6ヶ月)
- [ ] 高性能モデルの導入検討
- [ ] ファインチューニングの実施
- [ ] プロダクション環境への最適化

---

## 💡 **結論**

本ドキュメントで提示している改善案は、現在のデモ版RAGシステムを段階的に高度化するためのロードマップです。

### 開発方針
1. **段階的改善**: リスクを最小化しながら確実に精度向上
2. **互換性保持**: 既存機能を維持しながら新機能を追加
3. **評価駆動**: 定量的指標による継続的改善

### プロダクション展開への道筋
デモ版から本格運用システムへの発展において、本改善案は技術的実現可能性と費用対効果を考慮した実践的なアプローチだと考えています。

---

**文書作成日**: 2025年6月13日  
**バージョン**: 1.0  
**対象システム**: rag-chatbot-demo v1.0 (デモ版)
