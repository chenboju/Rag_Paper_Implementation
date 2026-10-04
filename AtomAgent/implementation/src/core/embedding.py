import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

def get_local_embeddings():
    """
    初始化並回傳運行於本地 GPU (RTX 4090) 的 Embedding 模型。
    針對 Qwen 系列模型進行了優化設定。
    """
    # 讀取配置
    # 預設值改為你指定的 Qwen 模型，防止 .env 沒讀到時出錯
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    device = os.getenv("EMBEDDING_DEVICE", "cuda")

    # 檢查 CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ Warning: CUDA is not available. Falling back to CPU.")
        device = "cpu"
    else:
        print(f"🚀 Initializing Embeddings [{model_name}] on {torch.cuda.get_device_name(0)}...")

    # 配置模型參數
    model_kwargs = {
        'device': device,
        'trust_remote_code': True  # [關鍵修改] Qwen 模型需要信任遠端代碼才能載入自定義架構
    }
    
    # 編碼參數
    encode_kwargs = {
        'normalize_embeddings': True, # 建議開啟
        'batch_size': 32              # RTX 4090 顯存很大，可以適當調大 batch_size 加速索引
    }

    try:
        # 初始化 LangChain 的 Embedding Wrapper
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            multi_process=False,
            show_progress=True
        )
    except OSError as e:
        print(f"\n❌ 模型載入失敗: {e}")
        print("💡 提示: 請確認 .env 中的模型名稱是否正確，或是否需要 `huggingface-cli login`。")
        raise e

    return embeddings

if __name__ == "__main__":
    try:
        emb_model = get_local_embeddings()
        
        test_text = "在 RTX 4090 上測試 Qwen 原子代理系統"
        vector = emb_model.embed_query(test_text)
        
        print(f"✅ Embedding generated for: '{test_text}'")
        print(f"✅ Vector dimension: {len(vector)}")
        # print(f"✅ Running on device: {emb_model.client.device}")  <-- 移除這行
        print("🎉 System Check Passed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")