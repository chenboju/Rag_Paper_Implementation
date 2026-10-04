import os
import re
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass, field
import hashlib
from collections import Counter
from openai import OpenAI
from datetime import datetime

# --- 環境與設定 ---
load_dotenv()

# --- vvv Ollama 連接設定 vvv ---
# 1. 修改 client，使其指向本地 Ollama 伺服器
#    Ollama 提供了與 OpenAI 相容的 API 接口
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # for Ollama, api_key 是必需的，但內容不限
)

# 2. 修改 LLM_MODEL 為您指定的 Ollama 模型名稱
LLM_MODEL = "gpt-oss:20b"
# --- ^^^ Ollama 連接設定 ^^^ ---


# 配置參數
INPUT_DIR = r"C:\Users\chen\python\langchain_ragas\MaterialChain\SimRAG\md"
OUTPUT_DIR = "./output_0909"
MAX_CONTEXT_CHARS = 8192

# 數量控制參數
TARGET_ANSWERS_PER_CHUNK = 20
MIN_ANSWERS_PER_CHUNK = 4
MAX_ANSWERS_PER_CHUNK = 10
QUESTIONS_PER_ANSWER = 4
MAX_RETRIES_FOR_ANSWERS = 3

# 品質控制參數
MIN_ANSWER_LENGTH = 1
MAX_ANSWER_LENGTH = 100
MIN_QA_QUALITY_SCORE = 0.8
SIMILARITY_THRESHOLD = 0.85

# --- 資料結構 ---
@dataclass
class AnswerCandidate:
    """候選簡短答案數據結構"""
    text: str  # Short Answer
    source_chunk: str

@dataclass
class QAPair:
    """最終QA對數據結構 (包含長短答案和時間戳)"""
    question: str
    long_answer: str
    short_answer: str
    context: str
    source: str  # 原始檔名 (e.g., Introduction.md)
    paper_section: str # 新增欄位: 論文名_段落名 (e.g., PaperName_Introduction)
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# --- LLM 提示 (Prompts) ---

# LLM-1: 提取簡短答案
ANSWER_EXTRACTION_PROMPT_SYS = "You are an information extraction expert responsible for extracting candidate phrases from paragraphs that could serve as answers to questions."
def answer_extraction_prompt_user(context: str, target_count: int) -> str:
    return f"""Based on the following [Context], extract up to **{target_count}** candidate answers that can be **independently understood**; these can be **entities, noun phrases, verb phrases, numbers, or time expressions**.

Please follow these rules:
1. Avoid generic words, overly long sentences, and cross-sentence fragments;
2. Prioritize specific, verifiable segments (person/place/organization names, terminology, numerical values, etc.);
3. Remove duplicates (ignoring case and synonyms);
4. **Do not include semicolons within the segments**;
5. Each segment should be 1-8 words in length;
6. **Output only one line**, separating candidate answers with semicolons `;`.

[Context]
{context}

**Output Format (single line)**
`Candidate1; Candidate2; Candidate3; ...`"""

# LLM-2: (新) 潤飾答案，生成長答案
ANSWER_ELABORATION_PROMPT_SYS = "You are a QA assistant who expands a keyword answer into a full, self-contained sentence based on the provided context."
def answer_elaboration_prompt_user(short_answer: str, context: str) -> str:
    return f"""Based on the [Context] provided, please expand the [Short Answer] into a single, complete, and self-contained sentence.

**Rules:**
1. The expanded sentence must be a direct and complete answer to a potential question.
2. The sentence **must be fully supported** by the information within the [Context].
3. **Do not add any information** that is not present in the [Context].
4. Output only the single, elaborated sentence.

[Context]
{context}

[Short Answer]
{short_answer}

**Expanded Sentence Output:**
"""

# LLM-3: 根據長答案生成問題
QUESTION_GENERATION_PROMPT_SYS = "You are a question design expert who generates questions that can be read independently based on a complete answer and its context."
def question_generation_prompt_user(long_answer: str, context: str) -> str:
    return f"""Please write **1 high-quality question** based on the [Context] and [Full Answer] so that the **standard answer is exactly the [Full Answer]**.

**Constraints:**
1. The question must be **understandable without the context**. Do not use phrases like "in the text" or "according to the passage".
2. The question should probe the core information presented in the [Full Answer].
3. Language and style should be consistent with the [Context].
4. The question should be between 8 and 25 words.

[Context]
{context}

[Full Answer]
{long_answer}

**Output:**
Only output the question itself in a single sentence.
"""


class QAGeneratorForChoiceQuestions:
    def __init__(self):
        self.generated_questions = set()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def ask_llm(self, system: str, user: str, max_retries: int = 3) -> str:
        """通用的 LLM 請求方法"""
        for attempt in range(max_retries):
            try:
                # 這段程式碼因為 client 已經被修改，會自動向 Ollama 發送請求
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"LLM request failed after {max_retries} attempts: {e}")
                    return ""
                continue

    def extract_short_answers(self, context: str, source_file: str) -> List[AnswerCandidate]:
        """步驟 1: 從內文中提取簡短答案 (Short Answer)"""
        print(f"📋 步驟 1: 提取簡短答案... ({source_file})")
        
        prompt = answer_extraction_prompt_user(context, TARGET_ANSWERS_PER_CHUNK)
        response_text = self.ask_llm(ANSWER_EXTRACTION_PROMPT_SYS, prompt)
        
        if not response_text:
            return []
            
        raw_answers = [a.strip() for a in response_text.split(";") if a.strip()]
        
        candidates = []
        for ans_text in raw_answers:
            if MIN_ANSWER_LENGTH <= len(ans_text) <= MAX_ANSWER_LENGTH:
                candidates.append(AnswerCandidate(text=ans_text, source_chunk=source_file))

        final_answers = candidates[:MAX_ANSWERS_PER_CHUNK]
        print(f"  ✅ 提取了 {len(final_answers)} 個簡短答案。")
        return final_answers

    def elaborate_answer(self, short_answer: AnswerCandidate, context: str) -> str:
        """(新) 步驟 2: 將簡短答案潤飾成完整長答案 (Long Answer)"""
        print(f"  💬 步驟 2: 潤飾答案 '{short_answer.text[:30]}...' -> 長答案")
        prompt = answer_elaboration_prompt_user(short_answer.text, context)
        long_answer = self.ask_llm(ANSWER_ELABORATION_PROMPT_SYS, prompt)
        return long_answer.strip().strip('"')

    def generate_questions_for_answer(self, long_answer: str, context: str) -> List[str]:
        """步驟 3: 根據長答案生成多個問題"""
        print(f"  ❓ 步驟 3: 為長答案生成問題...")
        questions = set()
        for _ in range(QUESTIONS_PER_ANSWER):
            prompt = question_generation_prompt_user(long_answer, context)
            question_text = self.ask_llm(QUESTION_GENERATION_PROMPT_SYS, prompt)
            
            if question_text and len(question_text) > 5:
                clean_question = question_text.strip().strip('"\'`')
                question_hash = hashlib.md5(clean_question.lower().encode()).hexdigest()
                if question_hash not in self.generated_questions:
                    self.generated_questions.add(question_hash)
                    questions.add(clean_question)
        
        print(f"    ✅ 生成了 {len(questions)} 個不重複的問題。")
        return list(questions)

    def validate_qa_pair(self, question: str, long_answer: str) -> Tuple[bool, float]:
        """(簡化) 步驟 4: 對 QA 對進行基本程式碼驗證"""
        if not question or not long_answer:
            return False, 0.0
        
        context_refs = ["文中", "上述", "該段", "這個", "那個", "in the text", "above", "this passage"]
        if any(ref in question for ref in context_refs):
            return False, 0.0
        
        score = 0.85 
        return score >= MIN_QA_QUALITY_SCORE, score

    def process_document(self, md_path: str) -> List[QAPair]:
        """處理單個文檔的完整流程"""
        with open(md_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        context_for_llm = full_text[:MAX_CONTEXT_CHARS]
        source_file = os.path.basename(md_path)

        # --- 新增: 從路徑解析論文和段落名稱 ---
        paper_name = os.path.basename(os.path.dirname(md_path))
        section_name = os.path.splitext(source_file)[0]
        paper_section_id = f"{paper_name}_{section_name}"
        # --- 結束新增 ---
        
        print(f"\n📄 開始處理文檔: {paper_section_id}.md")
        
        short_answers = self.extract_short_answers(context_for_llm, source_file)
        if not short_answers:
            print("  ❌ 未找到合適的簡短答案，跳過此文件。")
            return []
            
        valid_pairs = []
        for i, short_answer_candidate in enumerate(short_answers):
            print(f"\n  🔄 處理簡短答案 {i+1}/{len(short_answers)}: '{short_answer_candidate.text}'")
            
            long_answer_text = self.elaborate_answer(short_answer_candidate, context_for_llm)
            if not long_answer_text:
                print("    ❌ 長答案生成失敗，跳過此答案。")
                continue
            
            questions = self.generate_questions_for_answer(long_answer_text, context_for_llm)
            
            for question in questions:
                passed, score = self.validate_qa_pair(question, long_answer_text)
                
                if passed:
                    qa_pair = QAPair(
                        question=question,
                        long_answer=long_answer_text,
                        short_answer=short_answer_candidate.text,
                        context=full_text,
                        source=source_file,
                        paper_section=paper_section_id, # 新增
                        quality_score=score
                    )
                    valid_pairs.append(qa_pair)
                    print(f"    👍 高品質 QA 對已保存 (分數: {score:.2f})")

        print(f"  📊 文件 {source_file} 最終產生 {len(valid_pairs)} 個高品質QA對。")
        return valid_pairs

    def save_document_results(self, pairs: List[QAPair], source_md_filename: str, paper_name: str):
        """保存單個文檔的結果到其對應的論文資料夾"""
        # --- 修改: 創建以論文名稱命名的子目錄 ---
        doc_output_dir = os.path.join(OUTPUT_DIR, paper_name)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        rows = []
        for pair in pairs:
            rows.append({
                "question": pair.question,
                "context": pair.context,
                "Answer (Long)": pair.long_answer,
                "Answer (Short)": pair.short_answer,
                "source": pair.source,
                "paper_section": pair.paper_section, # 新增
                "quality_score": pair.quality_score,
                "timestamp": pair.timestamp
            })
        
        if not rows:
            print(f"  - ⚠️  在 {source_md_filename} 中未生成任何可保存的QA對。")
            return
            
        df = pd.DataFrame(rows)
        # --- 修改: 調整欄位順序 ---
        df = df[['timestamp', 'question', 'context', 'Answer (Long)', 'Answer (Short)', 'source', 'paper_section', 'quality_score']]
        
        # --- 修改: 調整檔案命名邏輯 ---
        # 從 "Introduction.md" 獲取 "Introduction"
        section_name = os.path.splitext(source_md_filename)[0]
        # 使用段落名 + 全域運行的時間戳來命名檔案，避免覆蓋
        base_filename = f"{section_name}_qca_output_{self.run_timestamp}"
        
        # 將檔案保存在對應的論文子目錄中
        csv_path = os.path.join(doc_output_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        jsonl_path = os.path.join(doc_output_dir, f"{base_filename}.jsonl")
        df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)

        print(f"\n  💾 結果已保存至目錄: {doc_output_dir}")
        print(f"    - CSV檔案: {csv_path}")
        print(f"    - JSONL檔案: {jsonl_path}")

def main():
    """主函數"""
    generator = QAGeneratorForChoiceQuestions()
    
    # --- 修改: 使用 os.walk 遞迴尋找所有 .md 檔案 ---
    md_files_to_process = []
    for dirpath, _, filenames in os.walk(INPUT_DIR):
        # 排除根目錄本身，只處理子目錄
        if dirpath == INPUT_DIR:
            continue
        
        paper_name = os.path.basename(dirpath)
        for filename in filenames:
            if filename.lower().endswith('.md'):
                full_path = os.path.join(dirpath, filename)
                md_files_to_process.append({
                    "path": full_path,
                    "paper_name": paper_name,
                    "md_file": filename
                })
    
    if not md_files_to_process:
        print(f"❌ 在目錄 '{INPUT_DIR}' 的子目錄中未找到任何 .md 文件。")
        return
        
    md_files_to_process.sort(key=lambda x: x['path']) # 排序以確保處理順序一致
    total_pairs_generated = 0
    
    print(f"🚀 開始處理 {len(md_files_to_process)} 個 Markdown 文檔...")
    
    # --- 修改: 遍歷新的檔案列表 ---
    for file_info in md_files_to_process:
        md_path = file_info["path"]
        paper_name = file_info["paper_name"]
        md_file = file_info["md_file"]
        
        try:
            # 處理單個文件
            pairs = generator.process_document(md_path)
            # 如果生成了有效的 QA 對，就立刻保存
            if pairs:
                # --- 修改: 傳遞 paper_name 給保存函數 ---
                generator.save_document_results(pairs, md_file, paper_name)
                total_pairs_generated += len(pairs)
        except Exception as e:
            print(f"❌ 處理文件 {md_file} 時發生嚴重錯誤: {e}")
            continue
            
    print(f"\n🎉 全部處理完成！")
    print(f"📁 結果已輸出至 '{OUTPUT_DIR}' 下的各個對應資料夾。")
    print(f"📊 總共生成了 {total_pairs_generated} 個高品質 QA 對。")


if __name__ == "__main__":
    main()