import os
import re
import time
import sys
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# NEW: 為了在本地運行模型，需要匯入 transformers 和 torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# ==============================================================================
# CONFIG: 所有設定請在此處集中管理
# ==============================================================================
class Config:
    """
    主設定檔：所有路徑、模型、模式開關、API參數都在此設定。
    """
    # --- 1. 檔案與路徑設定 ---
    DATA_PATH = "./csv/merged.csv"  # <--- 指定您的 QA 測試集檔案路徑

    # --- 2. 執行模式設定 ---
    NUM_SAMPLE = None            # 測試樣本數, None=全量
    DEBUG_MODE = True            # 是否開啟除錯模式 (會印出錯誤題目的詳細資訊)
    USE_TWO_STEP = True          # 是否使用 "推理+提取" 兩步驟流程

    # --- 3. RAG 模式設定 ---
    USE_RAG = False               # ⭐ 關鍵開關：True=RAG模式, False=純LLM模式
    RAG_TOP_K = 5                # RAG 檢索的文檔數量

    # --- 4. 模型設定 ---
    # MODIFIED: 直接使用 Hugging Face Hub 上的模型名稱 (Repository ID)
    # 程式會自動處理下載與快取
    MODEL_NAME_REASONING = "Qwen/Qwen3-8B"
    
    MODEL_NAME_EXTRACTION = "gpt-4.1-mini"  # 提取模型保持不變

    # MODIFIED: 本地模型載入設定
    DEVICE_MAP = "auto"
    TORCH_DTYPE = torch.bfloat16 

    # --- 5. VectorDB (Qdrant) 設定 ---
    QDRANT_URL = "http://localhost:6333"
    EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    COLLECTION_NAME = "genQA_qwen3_0_6b_RAGtest"

    # --- 6. CSV 檔案欄位名稱對應 ---
    class CsvColumns:
        QUESTION = "Question"
        CORRECT_ANSWER = "Correct Answer"
        ALL_OPTIONS = "all_option"

    # --- 7. 其他設定 ---
    DEFAULT_SUBJECT = "Abstract"

# ==============================================================================
# 主程式
# ==============================================================================

def print_config():
    """印出當前執行設定"""
    print("="*50)
    print(f"🤖 執行設定：")
    print(f"   - 測試檔案: {Config.DATA_PATH}")
    print(f"   - 樣本數: {'全量' if Config.NUM_SAMPLE is None else Config.NUM_SAMPLE}")
    # MODIFIED: 更新輸出的模型來源為 Hugging Face Hub ID
    print(f"   - 推理模型 (HF Hub): {Config.MODEL_NAME_REASONING}")
    print(f"   - 提取模型 (OpenAI): {Config.MODEL_NAME_EXTRACTION}")
    print(f"🔧 模式：")
    print(f"   - 除錯模式: {'開啟' if Config.DEBUG_MODE else '關閉'}")
    print(f"   - 兩步驟模式: {'開啟' if Config.USE_TWO_STEP else '關閉'}")
    print(f"   - RAG 模式: {'開啟' if Config.USE_RAG else '關閉'}")
    if Config.USE_RAG:
        print(f"📚 RAG 設定：")
        print(f"   - Embedding: {Config.EMB_MODEL}")
        print(f"   - Collection: {Config.COLLECTION_NAME}")
        print(f"   - Top-K: {Config.RAG_TOP_K}")
    print("="*50)

# --- 1. 讀取API Key ---
load_dotenv()
# MODIFIED: 重新加入 HF Token，用於首次下載或訪問需要權限的模型
API_KEY_HUGGINGFACE = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
API_KEY_OPENAI = os.environ.get("OPENAI_API_KEY")

def check_api_keys():
    if not API_KEY_HUGGINGFACE:
        print("⚠️ 警告: 未設定 HUGGINGFACEHUB_API_TOKEN。公開模型仍可下載，但建議設定以避免速率限制。")
    if not API_KEY_OPENAI:
        print("❌ 未設定 OPENAI_API_KEY，請於 .env 或環境變數中設置。")
        sys.exit(1)

# --- 2. LLM設定 ---
def initialize_llms():
    # MODIFIED: 推理模型改為從 HF Hub 自動下載/快取
    print(f"🔧 正在從 Hugging Face Hub 載入模型: {Config.MODEL_NAME_REASONING}...")
    print(f"   (如果本地快取不存在，將會自動下載)")
    print(f"   - 資料型態: {Config.TORCH_DTYPE}")
    print(f"   - 設備: {Config.DEVICE_MAP}")
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME_REASONING,
        token=API_KEY_HUGGINGFACE
    )

    # 載入模型 (無量化)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME_REASONING,
        device_map=Config.DEVICE_MAP,
        torch_dtype=Config.TORCH_DTYPE,
        token=API_KEY_HUGGINGFACE
    )
    
    # 建立 transformers pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1500,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    # 用 LangChain 包裝起來
    llm_reasoning = HuggingFacePipeline(pipeline=text_generation_pipeline)
    print(f"✅ 成功初始化本地模型")

    print("🔧 正在初始化 OpenAI 提取模型...")
    llm_extraction = ChatOpenAI(
        openai_api_key=API_KEY_OPENAI,
        model=Config.MODEL_NAME_EXTRACTION,
        temperature=0.7,
        max_tokens=512
    )
    print(f"✅ 成功初始化 OpenAI 模型")
    
    return llm_reasoning, llm_extraction, tokenizer

# --- 後續所有函式均無需修改 ---

def get_prompts():
    rag_reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a top expert in materials science, answering a multiple choice question based on the provided reference materials.\n\nRetrieved Reference Materials:\n{context}\n\nPlease follow these steps for your reasoning and answer:\n1. Carefully read and analyze the above retrieved materials to find the answer to the question.\n2. For each option, analyze step by step, citing relevant reference content.\n3. If the provided materials are insufficient, clearly state so and rely on your own expertise.\n4. After your reasoning, write your final answer on a new line using the exact format: FINAL_ANSWER: (X)"),
        ("user", "Question: {question}\nOptions: {options}")
    ])
    non_rag_reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a top expert in materials science, answering a multiple choice question.\n\nPlease follow these steps for your reasoning and answer:\n1. Carefully read the question and all options.\n2. Analyze each option step by step using principles from materials science.\n3. Justify why each option is correct or incorrect.\n4. After completing your reasoning, write your final answer on a new line, using the exact format: FINAL_ANSWER: (X)"),
        ("user", "Question: {question}\nOptions: {options}")
    ])
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are given a reasoning process that ends with a line in the format FINAL_ANSWER: ...\nYour task is to extract ONLY the answer that appears immediately after FINAL_ANSWER:.\n- Do NOT provide any explanation or extra text.\n- Output ONLY the content found after FINAL_ANSWER:.\nIf no line starting with FINAL_ANSWER: is found, output: [EMPTY]"),
        ("user", "Previous reasoning:\n{reasoning}")
    ])
    return rag_reasoning_prompt, non_rag_reasoning_prompt, extraction_prompt

def create_chains(llm_reasoning, llm_extraction, prompts, tokenizer):
    rag_prompt, non_rag_prompt, ext_prompt = prompts

    def apply_chat_template(prompt_value):
        messages = [{"role": msg.type, "content": msg.content} for msg in prompt_value.to_messages()]
        if messages[0]["role"] == "system":
            messages = [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user", "content": messages[1]["content"]}
            ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    reasoning_chain_rag = rag_prompt | RunnableLambda(apply_chat_template) | llm_reasoning | StrOutputParser()
    reasoning_chain_no_rag = non_rag_prompt | RunnableLambda(apply_chat_template) | llm_reasoning | StrOutputParser()
    extraction_chain = ext_prompt | llm_extraction | StrOutputParser()
    return reasoning_chain_rag, reasoning_chain_no_rag, extraction_chain

def initialize_rag_components():
    if not Config.USE_RAG:
        print("🚀 純LLM模式，跳過RAG組件初始化")
        return None
    print("🔧 正在初始化RAG組件...")
    try:
        emb_model = HuggingFaceEmbeddings(model_name=Config.EMB_MODEL)
        client = QdrantClient(url=Config.QDRANT_URL)
        print(f"✅ 成功載入Embedding模型: {Config.EMB_MODEL}")
        collections = client.get_collections().collections
        all_names = [col.name for col in collections]
        if Config.COLLECTION_NAME not in all_names:
            print(f"❌ 找不到指定 collection：{Config.COLLECTION_NAME}")
            print(f"   現有 collections: {all_names}")
            sys.exit(2)
        points_count = client.count(collection_name=Config.COLLECTION_NAME, exact=False).count
        print(f"✅ VectorDB '{Config.COLLECTION_NAME}' 連線成功，總 Points 數：{points_count}")
        vectorstore = Qdrant(client=client, collection_name=Config.COLLECTION_NAME, embeddings=emb_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RAG_TOP_K})
        print("✅ RAG組件初始化完成")
        return retriever
    except Exception as e:
        print(f"❌ RAG初始化失敗: {e}")
        print("💡 請檢查 Qdrant Docker 是否啟動且可連線。")
        sys.exit(1)

def ask(chain, inputs: dict, retries=3, delay=30) -> str:
    try:
        return chain.invoke(inputs)
    except Exception as e:
        return f"ERROR: {e}"

def extract_final_answer(response_text, from_extraction=True):
    if from_extraction:
        response = str(response_text).strip()
        return response.split('\n')[0].strip() if response and response != "[EMPTY]" else "[EMPTY]"
    else:
        match = re.search(r'FINAL_ANSWER:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip().split('\n')[0].strip() if match else "[EMPTY]"

def check_correctness(gold_answer_text, model_final_answer, all_options_str):
    if model_final_answer == "[EMPTY]" or model_final_answer.startswith("ERROR:"):
        return False
    options = re.findall(r'\(([A-D])\)\s*(.*?)\s*(?=\(|$)', all_options_str)
    gold_option_letter = ''
    for letter, text in options:
        if text.strip() == gold_answer_text.strip():
            gold_option_letter = letter
            break
    if not gold_option_letter:
        if Config.DEBUG_MODE: print(f"🔍 找不到標準答案 '{gold_answer_text}' 在選項 '{all_options_str}' 中的對應字母")
        return False
    model_option_letter = ''.join(sorted(list(set(re.findall(r'[A-D]', str(model_final_answer).upper())))))
    correct = (gold_option_letter == model_option_letter)
    if Config.DEBUG_MODE and not correct:
        print(f"   - 答案比對: 標準='({gold_option_letter})' vs 模型='({model_option_letter})' (原始='{model_final_answer}')")
    return correct

def load_dataset():
    try:
        df_full = pd.read_csv(Config.DATA_PATH, encoding='utf-8')
    except Exception:
        df_full = pd.read_csv(Config.DATA_PATH, encoding='latin1')
    if Config.NUM_SAMPLE and Config.NUM_SAMPLE > 0 and Config.NUM_SAMPLE < len(df_full):
        df = df_full.sample(n=Config.NUM_SAMPLE, random_state=42).reset_index(drop=True)
    else:
        df = df_full.reset_index(drop=True)
        Config.NUM_SAMPLE = len(df)
    print(f"🚀 成功載入測試集，將執行 {len(df)} 題")
    return df

def save_and_analyze_results(results, context_logs):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = re.sub(r'[^A-Za-z0-9]+', '_', Config.MODEL_NAME_REASONING)
    method_suffix = "RAG" if Config.USE_RAG else "LLM_Only"
    save_dir = f"results_{model_short_name}_{ts}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📁 輸出檔案將存於資料夾: {save_dir}/")
    df_res = pd.DataFrame(results)
    df_res["正確"] = df_res["正確"].fillna(False).astype(int)
    base_filename = f"Results_{method_suffix}_{Config.NUM_SAMPLE}samples_{ts}"
    csv_path = os.path.join(save_dir, f"{base_filename}.csv")
    df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📄 評測詳細結果 (CSV) 已儲存至: {csv_path}")
    json_path = os.path.join(save_dir, f"{base_filename}.json")
    df_res.to_json(json_path, orient="records", force_ascii=False, indent=4)
    print(f"📄 評測詳細結果 (JSON) 已儲存至: {json_path}")
    if Config.USE_RAG and context_logs:
        context_json_path = os.path.join(save_dir, f"Context_Log_{ts}.json")
        with open(context_json_path, 'w', encoding='utf-8') as f:
            json.dump(context_logs, f, ensure_ascii=False, indent=4)
        print(f"📄 RAG 檢索內容 (JSON) 已儲存至: {context_json_path}")
    acc_overall = df_res["正確"].mean()
    print("\n" + "="*50)
    print(f"📊 最終評測結果")
    print(f"   - 方法: {method_suffix}")
    print(f"   - 整體準確率: {acc_overall:.2%} ({df_res['正確'].sum()}/{len(df_res)})")
    print("="*50)

def main():
    print_config()
    check_api_keys()
    df = load_dataset()
    llm_reasoning, llm_extraction, tokenizer = initialize_llms()
    retriever = initialize_rag_components()
    prompts = get_prompts()
    reasoning_chain_rag, reasoning_chain_no_rag, extraction_chain = create_chains(llm_reasoning, llm_extraction, prompts, tokenizer)
    results = []
    context_logs = []
    method_name = "RAG-TwoStep" if Config.USE_RAG else "TwoStep"
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"評測進度 ({method_name})"):
        try:
            q = row[Config.CsvColumns.QUESTION]
            gold = str(row[Config.CsvColumns.CORRECT_ANSWER]).strip()
            all_options = row[Config.CsvColumns.ALL_OPTIONS]
            subj = str(row.get(getattr(Config.CsvColumns, 'SUBJECT', 'subject'), Config.DEFAULT_SUBJECT))
            if Config.USE_RAG and retriever:
                retrieved_docs: list[Document] = retriever.invoke(q)
                context_for_prompt = "\n\n".join([doc.page_content for doc in retrieved_docs])
                context_log_item = {"question_index": idx, "question": q, "retrieved_chunks": [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]}
                context_logs.append(context_log_item)
                reasoning_response = ask(reasoning_chain_rag, {"question": q, "options": all_options, "context": context_for_prompt})
            else:
                reasoning_response = ask(reasoning_chain_no_rag, {"question": q, "options": all_options})
            if "ERROR:" in reasoning_response:
                final_answer = reasoning_response
                extraction_response = "推理失敗"
            elif Config.USE_TWO_STEP:
                extraction_response = ask(extraction_chain, {"reasoning": reasoning_response})
                final_answer = extract_final_answer(extraction_response, from_extraction=True)
            else:
                extraction_response = "未啟用兩步驟"
                final_answer = extract_final_answer(reasoning_response, from_extraction=False)
            correct = check_correctness(gold, final_answer, all_options)
            results.append({"subject": subj, "問題": q, "標準答案": gold, "模型答案": final_answer, "正確": correct, "推理過程": reasoning_response, "提取回應": extraction_response})
            if Config.DEBUG_MODE and not correct:
                print(f"\n❌ 錯誤題目 #{idx+1}: {q[:80]}...")
                print(f"   - 標準答案: {gold}")
        except Exception as e:
            print(f"⚠️ 處理題目 {idx+1} 時發生嚴重錯誤: {e}")
            results.append({"問題": q, "模型答案": f"ERROR: {e}", "正確": False})
    save_and_analyze_results(results, context_logs)
    print(f"\n✅ 評測完成！")

if __name__ == "__main__":
    main()