import os
import json
import pandas as pd
from openai import OpenAI
import random
from tqdm import tqdm
import re
from typing import List, Dict, Any, Optional

# ==============================================================================
#  使用者設定區域
# ==============================================================================
QA_SOURCE_DIR = r"C:\Users\chen\python\langchain_ragas\MaterialChain\SimRAG\md\filtered_ragas_results_source_structured_0911"
CONTEXT_SOURCE_DIR = r"C:\Users\chen\python\langchain_ragas\MaterialChain\SimRAG\rag_QA_test"
FINAL_OUTPUT_ROOT_DIR = r"./final_mcq_structured_output"

# --- vvv Ollama 連接設定 vvv ---
try:
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )
    LLM_MODEL = "gpt-oss:20b"
    client.models.list()
    print("✅ 成功連接到本地 Ollama 伺服器。")
except Exception as e:
    print(f"❌ 無法連接到本地 Ollama 伺服器 (http://localhost:11434)。請確保其正在運行。")
    print(f"   錯誤訊息: {e}")
    exit()
# --- ^^^ Ollama 連接設定 ^^^ ---
# ==============================================================================

def find_context_for_qa_file(qa_json_path: str) -> Optional[str]:
    """根據 QA JSON 檔案的路徑，推斷並讀取對應的 Markdown 上下文檔案。"""
    try:
        path_parts = qa_json_path.split(os.sep)
        paper_folder_name = path_parts[-2]
        filename = os.path.basename(qa_json_path)
        match = re.search(r'ragas_eval_(.+?)_qca_output', filename)
        if not match: return None
        section_name = match.group(1)
        
        context_md_path = os.path.join(CONTEXT_SOURCE_DIR, paper_folder_name, f"{section_name}.md")
        if not os.path.exists(context_md_path):
            context_md_path = os.path.join(CONTEXT_SOURCE_DIR, paper_folder_name, f"{section_name.upper()}.md")
            if not os.path.exists(context_md_path): return None

        with open(context_md_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None

def generate_distractors(context: str, question: str, correct_answer: str) -> List[str]:
    """使用兩段式生成策略，並注入「簡短答案」概念來獲取誘答選項。"""
    distractors = []

    # --- 第一次嘗試：JSON 模式 (注入 Short Answer 概念) ---
    try:
        # ==========================================================================
        #  ✨✨✨ 核心修改 1：在 JSON Prompt 中加入 Short Answer 風格要求 ✨✨✨
        # ==========================================================================
        prompt_json = f"""
        Based on the CONTEXT provided, generate three **short, concise** incorrect but plausible distractors for the QUESTION.
        **Style requirement: The distractors should be entities, technical terms, noun phrases, or numerical values, similar in style to a short, keyword-style answer, NOT a full sentence.**

        The CORRECT ANSWER is "{correct_answer}". Your distractors must NOT be the correct answer.
        All distractors MUST come from the information within the CONTEXT.
        Output ONLY a single JSON array of three strings.

        ---
        CONTEXT:
        {context}
        ---
        QUESTION:
        {question}
        ---
        JSON Array of three short, keyword-style distractors:
        """
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
                {"role": "user", "content": prompt_json}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            timeout=30.0
        )
        distractor_text = response.choices[0].message.content
        json_str = distractor_text[distractor_text.find('['):distractor_text.rfind(']')+1]
        if json_str:
            parsed_json = json.loads(json_str)
            distractors = [str(d).strip() for d in parsed_json if str(d).strip()]
    except Exception:
        pass

    # --- 第二次嘗試：純文字模式 (如果第一次失敗, 同樣注入 Short Answer 概念) ---
    if len(distractors) < 3:
        try:
            # ==========================================================================
            #  ✨✨✨ 核心修改 2：在純文字 Prompt 中加入 Short Answer 風格要求 ✨✨✨
            # ==========================================================================
            prompt_text = f"""
            Read the CONTEXT and QUESTION. The CORRECT ANSWER is "{correct_answer}".
            Provide three plausible but incorrect answer choices based on the context.
            **IMPORTANT STYLE RULE: Each choice must be a short, concise phrase (like a keyword, entity, or technical term), NOT a full sentence.**
            List each choice on a new line. Do not use numbers or bullet points.

            ---
            CONTEXT:
            {context}
            ---
            QUESTION:
            {question}

            ---
            Three incorrect, short, keyword-style choices, each on a new line:
            """
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.75,
                timeout=30.0
            )
            distractor_text = response.choices[0].message.content
            distractors = [line.strip() for line in distractor_text.split('\n') if line.strip()]
        except Exception:
            return []
            
    return distractors[:3]

def main():
    """主執行函數"""
    print("\n--- 步驟 1: 正在掃描 QA 來源目錄 ---")
    qa_json_files = [os.path.join(r, f) for r, _, fs in os.walk(QA_SOURCE_DIR) for f in fs if f.endswith('.json')]
    if not qa_json_files:
        print(f"❌ 在目錄 '{QA_SOURCE_DIR}' 中未找到任何 .json 檔案。")
        return
    print(f"✅ 找到 {len(qa_json_files)} 個 QA JSON 檔案。")

    print("\n--- 步驟 2: 開始處理檔案並生成選擇題 ---")
    total_mcqs_generated = 0
    
    for qa_path in tqdm(qa_json_files, desc="檔案處理進度", unit="file"):
        context = find_context_for_qa_file(qa_path)
        if not context: continue
            
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
        except Exception:
            continue
            
        mcq_for_this_file = []
        for item in qa_data:
            question, answer = item.get('user_input'), item.get('reference')
            if not (question and answer): continue
            
            print("\n" + "="*80)
            relative_qa_path = os.path.relpath(qa_path, QA_SOURCE_DIR)
            print(f"📄 正在處理檔案: {relative_qa_path}")
            print(f"❓ 問題: {question}")
            print(f"✅ 正確答案: {answer}")
            print("🧠 正在呼叫 LLM 生成「簡短答案風格」的誘答選項...")

            distractors = generate_distractors(context, question, answer)

            if len(distractors) == 3:
                print(f"   👍 成功生成誘答選項: {distractors}")
                options = [answer] + distractors
                random.shuffle(options)
                
                mcq_item = {'Question': question, 'Correct Answer': answer}
                for i, option in enumerate(options):
                    mcq_item[f'Option {chr(65+i)}'] = option
                mcq_for_this_file.append(mcq_item)
            else:
                print(f"   ⚠️  警告: 未能生成足夠的誘答選項({len(distractors)}/3)，跳過此題。")
        
        if mcq_for_this_file:
            relative_path = os.path.relpath(os.path.dirname(qa_path), QA_SOURCE_DIR)
            output_dir = os.path.join(FINAL_OUTPUT_ROOT_DIR, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            
            base_filename_match = re.search(r'ragas_eval_(.+?)_qca_output', os.path.basename(qa_path))
            new_base_name = f"mcq_{base_filename_match.group(1)}" if base_filename_match else f"mcq_{os.path.splitext(os.path.basename(qa_path))[0]}"

            mcq_df = pd.DataFrame(mcq_for_this_file).drop_duplicates(subset=['Question'])
            
            def format_all_options(row):
                opts = [f"({chr(65+i)}){row[f'Option {chr(65+i)}']}" for i in range(4) if f'Option {chr(65+i)}' in row and pd.notna(row[f'Option {chr(65+i)}'])]
                return " ".join(opts)
            mcq_df['all_option'] = mcq_df.apply(format_all_options, axis=1)
            
            column_order = ['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'all_option']
            mcq_df = mcq_df.reindex(columns=column_order)

            mcq_df.to_csv(os.path.join(output_dir, f"{new_base_name}.csv"), index=False, encoding='utf-8-sig')
            mcq_df.to_json(os.path.join(output_dir, f"{new_base_name}.json"), orient='records', indent=4, force_ascii=False)
            
            total_mcqs_generated += len(mcq_df)

    print("\n\n" + "="*80)
    print("🎉🎉🎉 恭喜！所有任務已全部完成！ 🎉🎉🎉")
    print(f"總共生成了 {total_mcqs_generated} 道高品質的選擇題。")
    print(f"所有結果已儲存至根目錄: '{FINAL_OUTPUT_ROOT_DIR}'")
    print("="*80)

if __name__ == "__main__":
    main()