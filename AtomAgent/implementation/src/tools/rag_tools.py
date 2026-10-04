import os
import uuid
import logging
from typing import List, Optional
from dotenv import load_dotenv

# 強制載入環境變數
load_dotenv()

from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# --- 設定 ---
DOCS_DIR = os.path.join(os.getenv("DATA_DIR", "./data"), "docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "atom_agents_knowledge"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64  # [新增功能] 批次寫入的大小，每次處理 64 個 chunk

# 設定 Logger
logger = logging.getLogger(__name__)

def get_embedding_model():
    """初始化 Embedding 模型"""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

def _initialize_vector_db():
    """
    初始化並連接 Qdrant 資料庫。
    包含：資料量檢查、自動維度偵測、批次寫入功能。
    """
    client = QdrantClient(url=QDRANT_URL)
    embeddings = get_embedding_model()

    # 1. 檢查 Collection 是否存在與資料量
    collection_exists = client.collection_exists(COLLECTION_NAME)
    doc_count = 0
    
    if collection_exists:
        doc_count = client.count(COLLECTION_NAME).count

    # 2. 根據您的需求執行邏輯判斷
    if doc_count > 0:
        # 情境 A: 資料庫有資料
        print(f"資料庫 \"{COLLECTION_NAME}\" 總共 {doc_count} 筆資料")
        return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    
    else:
        # 情境 B: 資料庫不存在 或 資料量 <= 0
        print(f"資料庫 \"{COLLECTION_NAME}\" 沒有資料")
        print(f"⚙️ 準備開始建立索引 (Batch Size: {BATCH_SIZE})...")

        # 若存在但為空，先刪除以確保設定乾淨
        if collection_exists:
            client.delete_collection(COLLECTION_NAME)

        # [新增功能] 維度自動化：先試跑一個 embedding 抓取維度
        print("📏 Detecting embedding dimension...")
        test_emb = embeddings.embed_query("test")
        emb_dim = len(test_emb)
        print(f"   -> Detected Dimension: {emb_dim}")

        # 建立 Collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE),
        )

        # 讀取 MD 資料
        print(f"📂 Loading Markdown files from {DOCS_DIR}...")
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
            
        loader = DirectoryLoader(
            DOCS_DIR, 
            glob="**/*.md", 
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'} # 強制 UTF-8
        )
        documents = loader.load()

        if not documents:
            print("⚠️ Warning: No Markdown files found. Created empty collection.")
            return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

        # 切分文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        print(f"🧩 Processing {len(splits)} chunks with metadata...")
        
        # 處理 Metadata
        processed_docs = []
        for doc in splits:
            chunk_id = str(uuid.uuid4())
            source_path = doc.metadata.get("source", "")
            file_name = os.path.basename(source_path) if source_path else "unknown.md"
            
            doc.metadata = {
                "id": chunk_id,
                "filename": file_name,
                "source": source_path,
            }
            processed_docs.append(doc)

        # 初始化 VectorStore (但不立刻加入文件)
        vectorstore = QdrantVectorStore(
            client=client, 
            collection_name=COLLECTION_NAME, 
            embedding=embeddings
        )

        # [新增功能] 批次寫入 (Batch Ingestion)
        total_chunks = len(processed_docs)
        print(f"🚀 Starting batch ingestion ({total_chunks} chunks)...")
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = processed_docs[i : i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            print(f"   -> Inserted batch {i} to {min(i + BATCH_SIZE, total_chunks)} / {total_chunks}")

        print("✅ Ingestion complete.")
        return vectorstore

# 初始化
try:
    vectorstore = _initialize_vector_db()
except Exception as e:
    print(f"❌ Critical Error connecting to Qdrant: {e}")
    vectorstore = None

@tool("retrieve_knowledge")
def retrieve_knowledge(query: str) -> str:
    """
    Search the internal knowledge base (Markdown files) for scientific principles, 
    past research, or material properties.
    """
    if vectorstore is None:
        return "Error: Knowledge base unavailable."
    
    print(f"🔎 RAG Searching for: {query}")
    results = vectorstore.similarity_search(query, k=3)
    
    output_text = []
    for i, doc in enumerate(results):
        source_name = doc.metadata.get('filename', 'unknown')
        content = doc.page_content.strip()
        output_text.append(f"[Source {i+1}: {source_name}]\n{content}")
    
    return "\n\n".join(output_text)