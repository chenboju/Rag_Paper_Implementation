import os
import re
import time
import sys
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

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
    NUM_SAMPLE = None            # 測試樣本數, None=全量, 或設為數字 (例如 5) 進行快速測試
    DEBUG_MODE = True            # 是否開啟除錯模式 (會印出錯誤題目的詳細資訊)
    USE_TWO_STEP = True          # 是否使用 "推理+提取" 兩步驟流程

    # --- 3. RAG 模式設定 ---
    USE_RAG = True               # ⭐ 關鍵開關：True=RAG模式, False=純LLM模式
    RAG_TOP_K = 5                # RAG 檢索的文檔數量

    # --- 4. 模型設定 ---
    MODEL_NAME_REASONING = "openai/gpt-5"    # 用於推理思考的語言模型
    MODEL_NAME_EXTRACTION = "gpt-4.1-mini"   # 用於提取最終答案的語言模型 (建議使用高準確率模型)

    # --- 5. VectorDB (Qdrant) 設定 ---
    QDRANT_URL = "http://localhost:6333"
    EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    # !重要: 此 Collection 名稱必須與您上傳資料時使用的名稱完全一致
    COLLECTION_NAME = "genQA_qwen3_0_6b_RAGtest"

    # --- 6. CSV 檔案欄位名稱對應 ---
    class CsvColumns:
        """
        對應您 CSV 檔案中的欄位名稱，如果您的欄位名不同，請在此修改。
        """
        QUESTION = "Question"
        CORRECT_ANSWER = "Correct Answer"
        ALL_OPTIONS = "all_option"
        # 如果您的 CSV 有主題(subject)欄位，可以在此指定，否則會使用下面的預設值
        # SUBJECT = "your_subject_column_name"

    # --- 7. 其他設定 ---
    DEFAULT_SUBJECT = "Abstract" # 如果 CSV 中沒有主題欄位，則使用此預設值

# ==============================================================================
# 主程式 (通常您不需要修改以下內容)
# ==============================================================================

def print_config():
    """印出當前執行設定"""
    print("="*50)
    print(f"🤖 執行設定：")
    print(f"   - 測試檔案: {Config.DATA_PATH}")
    print(f"   - 樣本數: {'全量' if Config.NUM_SAMPLE is None else Config.NUM_SAMPLE}")
    print(f"   - 推理模型: {Config.MODEL_NAME_REASONING}")
    print(f"   - 提取模型: {Config.MODEL_NAME_EXTRACTION}")
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
API_KEY_GEMMA = os.environ.get("OPENROUTER_API_KEY")
API_KEY_OPENAI = os.environ.get("OPENAI_API_KEY")

def check_api_keys():
    if not API_KEY_GEMMA:
        print("❌ 未設定 OPENROUTER_API_KEY，請於 .env 或環境變數中設置。")
        sys.exit(1)
    if not API_KEY_OPENAI:
        print("❌ 未設定 OPENAI_API_KEY，請於 .env 或環境變數中設置。")
        sys.exit(1)

# --- 2. LLM設定 ---
def initialize_llms():
    llm_reasoning = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=API_KEY_GEMMA,
        model=Config.MODEL_NAME_REASONING,
        temperature=0.7,
    )
    llm_extraction = ChatOpenAI(
        openai_api_key=API_KEY_OPENAI,
        model=Config.MODEL_NAME_EXTRACTION,
        temperature=0.7,
        max_tokens=512
    )
    return llm_reasoning, llm_extraction

# --- 3. RAG組件初始化 ---
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
        
        # ⭐ 更新: 這個 retriever 現在會返回完整的 Document 物件 (包含 metadata)
        retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RAG_TOP_K})
        
        print("✅ RAG組件初始化完成")
        return retriever
        
    except Exception as e:
        print(f"❌ RAG初始化失敗: {e}")
        print("💡 請檢查 Qdrant Docker 是否啟動且可連線。")
        sys.exit(1)

# --- 4. Prompt設計與Chain的建立 ---
def get_prompts():
    rag_reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a top expert in materials science, answering a multiple choice question based on the provided reference materials.

Retrieved Reference Materials:
{context}

Please follow these steps for your reasoning and answer:
1. Carefully read and analyze the above retrieved materials to find the answer to the question.
2. For each option, analyze step by step, citing relevant reference content.
3. If the provided materials are insufficient, clearly state so and rely on your own expertise.
4. After your reasoning, write your final answer on a new line using the exact format: FINAL_ANSWER: (X)
"""),
        ("user", "Question: {question}\nOptions: {options}")
    ])

    non_rag_reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a top expert in materials science, answering a multiple choice question.

Please follow these steps for your reasoning and answer:
1. Carefully read the question and all options.
2. Analyze each option step by step using principles from materials science.
3. Justify why each option is correct or incorrect.
4. After completing your reasoning, write your final answer on a new line, using the exact format: FINAL_ANSWER: (X)
"""),
        ("user", "Question: {question}\nOptions: {options}")
    ])

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are given a reasoning process that ends with a line in the format FINAL_ANSWER: ...
Your task is to extract ONLY the answer that appears immediately after FINAL_ANSWER:.
- Do NOT provide any explanation or extra text.
- Output ONLY the content found after FINAL_ANSWER:.
If no line starting with FINAL_ANSWER: is found, output: [EMPTY]
"""),
        ("user", "Previous reasoning:\n{reasoning}")
    ])
    
    return rag_reasoning_prompt, non_rag_reasoning_prompt, extraction_prompt

def create_chains(llm_reasoning, llm_extraction, prompts):
    rag_prompt, non_rag_prompt, ext_prompt = prompts
    
    # RAG 或非 RAG 流程將在主迴圈中決定，這裡只建立基本的 chain
    reasoning_chain_rag = rag_prompt | llm_reasoning | StrOutputParser()
    reasoning_chain_no_rag = non_rag_prompt | llm_reasoning | StrOutputParser()
    extraction_chain = ext_prompt | llm_extraction | StrOutputParser()

    return reasoning_chain_rag, reasoning_chain_no_rag, extraction_chain

# --- 5. 核心執行與檢查函數 ---
def ask(chain, inputs: dict, retries=3, delay=30) -> str:
    for i in range(retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if ("rate_limit" in str(e).lower() or "429" in str(e)) and i < retries - 1:
                print(f"⚠️ 遇到速率限制，等待{delay}秒後重試...")
                time.sleep(delay)
            else: return f"ERROR: {e}"
    return "ERROR: Max retries reached"

def extract_final_answer(response_text, from_extraction=True):
    if from_extraction:
        response = str(response_text).strip()
        return response.split('\n')[0].strip() if response and response != "[EMPTY]" else "[EMPTY]"
    else: # from reasoning
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

# --- 6. 載入資料集 ---
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

# --- 7. 儲存與分析結果 ---
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
    
    # 儲存主要結果 (CSV)
    csv_path = os.path.join(save_dir, f"{base_filename}.csv")
    df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📄 評測詳細結果 (CSV) 已儲存至: {csv_path}")

    # 儲存主要結果 (JSON)
    json_path = os.path.join(save_dir, f"{base_filename}.json")
    df_res.to_json(json_path, orient="records", force_ascii=False, indent=4)
    print(f"📄 評測詳細結果 (JSON) 已儲存至: {json_path}")

    # ⭐ 新增: 儲存 Context Log (JSON)
    if Config.USE_RAG and context_logs:
        context_json_path = os.path.join(save_dir, f"Context_Log_{ts}.json")
        with open(context_json_path, 'w', encoding='utf-8') as f:
            json.dump(context_logs, f, ensure_ascii=False, indent=4)
        print(f"📄 RAG 檢索內容 (JSON) 已儲存至: {context_json_path}")

    # ⭐ 新增: 計算並儲存準確率摘要
    if not df_res.empty:
        # 按 'subject' 分組計算準確率
        accuracy_by_subject = df_res.groupby('subject')['正確'].agg(['sum', 'count']).reset_index()
        accuracy_by_subject['準確率'] = (accuracy_by_subject['sum'] / accuracy_by_subject['count'])
        accuracy_by_subject.rename(columns={'sum': '答對題數', 'count': '總題數', 'subject': '主題'}, inplace=True)
        
        # 計算總體準確率
        total_correct = df_res['正確'].sum()
        total_questions = len(df_res)
        overall_accuracy = (total_correct / total_questions) if total_questions > 0 else 0
        
        # 建立總計列
        overall_summary = pd.DataFrame([{
            '主題': '總計 (Overall)',
            '答對題數': total_correct,
            '總題數': total_questions,
            '準確率': overall_accuracy
        }])
        
        # 合併分組結果與總計
        accuracy_summary_df = pd.concat([accuracy_by_subject, overall_summary], ignore_index=True)
        
        # 格式化準確率欄位為百分比字串
        accuracy_summary_df['準確率'] = accuracy_summary_df['準確率'].apply(lambda x: f"{x:.2%}")
        
        # 儲存準確率摘要 CSV
        accuracy_csv_path = os.path.join(save_dir, f"Accuracy_Summary_{ts}.csv")
        accuracy_summary_df.to_csv(accuracy_csv_path, index=False, encoding='utf-8-sig')
        print(f"📄 評測準確率摘要 (CSV) 已儲存至: {accuracy_csv_path}")


    acc_overall = df_res["正確"].mean()
    print("\n" + "="*50)
    print(f"📊 最終評測結果")
    print(f"   - 方法: {method_suffix}")
    print(f"   - 整體準確率: {acc_overall:.2%} ({df_res['正確'].sum()}/{len(df_res)})")
    print("="*50)

# --- 8. 主評測迴圈 ---
def main():
    print_config()
    check_api_keys()
    
    df = load_dataset()
    llm_reasoning, llm_extraction = initialize_llms()
    retriever = initialize_rag_components()
    prompts = get_prompts()
    reasoning_chain_rag, reasoning_chain_no_rag, extraction_chain = create_chains(llm_reasoning, llm_extraction, prompts)
    
    results = []
    context_logs = []  # ⭐ 新增: 用於儲存 RAG 檢索內容
    method_name = "RAG-TwoStep" if Config.USE_RAG else "TwoStep"
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"評測進度 ({method_name})"):
        try:
            q = row[Config.CsvColumns.QUESTION]
            gold = str(row[Config.CsvColumns.CORRECT_ANSWER]).strip()
            all_options = row[Config.CsvColumns.ALL_OPTIONS]
            subj = str(row.get(getattr(Config.CsvColumns, 'SUBJECT', 'subject'), Config.DEFAULT_SUBJECT))

            # ⭐ 更新: RAG 流程與 Context 紀錄
            if Config.USE_RAG and retriever:
                retrieved_docs: list[Document] = retriever.invoke(q)
                context_for_prompt = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 記錄檢索到的 context 及其 metadata
                context_log_item = {
                    "question_index": idx,
                    "question": q,
                    "retrieved_chunks": [
                        {"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs
                    ]
                }
                context_logs.append(context_log_item)
                
                reasoning_response = ask(reasoning_chain_rag, {"question": q, "options": all_options, "context": context_for_prompt})
            else:
                # 執行非 RAG 流程
                reasoning_response = ask(reasoning_chain_no_rag, {"question": q, "options": all_options})

            # 步驟 2: 提取 (如果啟用)
            if "ERROR:" in reasoning_response:
                final_answer = reasoning_response
                extraction_response = "推理失敗"
            elif Config.USE_TWO_STEP:
                extraction_response = ask(extraction_chain, {"reasoning": reasoning_response})
                final_answer = extract_final_answer(extraction_response, from_extraction=True)
            else:
                extraction_response = "未啟用兩步驟"
                final_answer = extract_final_answer(reasoning_response, from_extraction=False)
            
            # 步驟 3: 檢查
            correct = check_correctness(gold, final_answer, all_options)

            results.append({
                "subject": subj, "問題": q, "標準答案": gold, "模型答案": final_answer,
                "正確": correct, "推理過程": reasoning_response, "提取回應": extraction_response
            })

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

