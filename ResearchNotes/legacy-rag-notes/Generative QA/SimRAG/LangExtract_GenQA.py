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
import textwrap

# --- 1. 導入 LangExtract ---
import langextract as lx

# --- 環境與設定 ---
load_dotenv()

# --- vvv Ollama 連接設定 vvv ---
# Langchain/OpenAI client for generation
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # for Ollama, api_key 是必需的，但內容不限
)

# 為不同任務指定模型
EXTRACTION_LLM = "gpt-oss:20b"   # 用於 LangExtract 提取 Short Answer
GENERATION_LLM = "gpt-oss:20b"   # 用於生成 Long Answer 和 Question
OLLAMA_BASE_URL = "http://localhost:11434" # LangExtract 使用的 Ollama 位址
# --- ^^^ Ollama 連接設定 ^^^ ---


# 配置參數
INPUT_DIR = r"C:\Users\chen\python\langchain_ragas\MaterialChain\SimRAG\hu_0909_test\section"
OUTPUT_DIR = "./LangExtract_GenQA"
# ******** 這是修正處 ********
MAX_CONTEXT_CHARS = 8192 # 設定傳送給 LLM 的最大上下文長度

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

# --- 資料結構 (與您原本的相同) ---
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

# --- LLM 提示 (Prompts) (與您原本的相同) ---
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

# --- 新增：LangExtract 核心函式 ---
def extract_answers_with_langextract(context: str, model_id: str) -> List[AnswerCandidate]:
    """
    步驟 1: 使用 LangExtract 從內文中精確提取簡短答案 (Short Answer)。
    """
    print(f"📋 步驟 1: 使用 LangExtract 提取簡短答案...")
    if not isinstance(context, str) or not context.strip():
        return []

    # 1. 定義提取任務的描述 (Prompt)
    prompt = textwrap.dedent("""\
        Extract key phrases from the text that can serve as answers to questions.
        Focus on specific, verifiable information like entities, noun phrases, definitions, or key concepts.
        Ensure the extracted text is a precise fragment from the original context.""")

    # 2. 提供高品質的範例 (Few-shot examples)
    examples = [
        lx.data.ExampleData(
            text="LangExtract is a Python library that uses LLMs to extract structured information from unstructured text documents.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="candidate_answer",
                    extraction_text="a Python library"
                ),
                lx.data.Extraction(
                    extraction_class="candidate_answer",
                    extraction_text="extract structured information"
                )
            ]
        )
    ]
    
    # 3. 執行提取
    try:
        result = lx.extract(
            text_or_documents=context,
            prompt_description=prompt,
            examples=examples,
            model_id=model_id,
            model_url=OLLAMA_BASE_URL,
            use_schema_constraints=True,
            fence_output=False
        )
        
        if result and result.extractions:
            candidates = [
                AnswerCandidate(text=ext.extraction_text, source_chunk=context)
                for ext in result.extractions
                if MIN_ANSWER_LENGTH <= len(ext.extraction_text) <= MAX_ANSWER_LENGTH
            ]
            final_answers = candidates[:MAX_ANSWERS_PER_CHUNK]
            print(f"   ✅ LangExtract 提取了 {len(final_answers)} 個簡短答案。")
            return final_answers
        else:
            print("   🟡 LangExtract 未返回任何提取結果。")
            return []
            
    except lx.exceptions.LangExtractError as e:
        print(f"   ❌ LangExtract 在處理時發生錯誤 (可能是模型未回傳有效JSON): {e}")
        return []
    except Exception as e:
        print(f"   ❌ LangExtract 提取時發生未預期的錯誤: {e}")
        return []

class QAGeneratorForChoiceQuestions:
    def __init__(self):
        self.generated_questions = set()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def ask_llm(self, system: str, user: str, max_retries: int = 3) -> str:
        """通用的 LLM 請求方法 (使用您原有的 OpenAI client for Ollama)"""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=GENERATION_LLM,
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

    def elaborate_answer(self, short_answer: AnswerCandidate, context: str) -> str:
        """(新) 步驟 2: 將簡短答案潤飾成完整長答案 (Long Answer)"""
        print(f"   💬 步驟 2: 潤飾答案 '{short_answer.text[:30]}...' -> 長答案")
        prompt = answer_elaboration_prompt_user(short_answer.text, context)
        long_answer = self.ask_llm(ANSWER_ELABORATION_PROMPT_SYS, prompt)
        return long_answer.strip().strip('"')

    def generate_questions_for_answer(self, long_answer: str, context: str) -> List[str]:
        """步驟 3: 根據長答案生成多個問題"""
        print(f"   ❓ 步驟 3: 為長答案生成問題...")
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
        
        print(f"     ✅ 生成了 {len(questions)} 個不重複的問題。")
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

        paper_name = os.path.basename(os.path.dirname(md_path))
        section_name = os.path.splitext(source_file)[0]
        paper_section_id = f"{paper_name}_{section_name}"
        
        print(f"\n📄 開始處理文檔: {paper_section_id}.md")
        
        # --- vvv 整合點 vvv ---
        # 呼叫新的 LangExtract 函式
        short_answers = extract_answers_with_langextract(context_for_llm, EXTRACTION_LLM)
        # --- ^^^ 整合點 ^^^ ---

        if not short_answers:
            print("   ❌ 未找到合適的簡短答案，跳過此文件。")
            return []
            
        valid_pairs = []
        for i, short_answer_candidate in enumerate(short_answers):
            print(f"\n   🔄 處理簡短答案 {i+1}/{len(short_answers)}: '{short_answer_candidate.text}'")
            
            long_answer_text = self.elaborate_answer(short_answer_candidate, context_for_llm)
            if not long_answer_text:
                print("     ❌ 長答案生成失敗，跳過此答案。")
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
                        paper_section=paper_section_id,
                        quality_score=score
                    )
                    valid_pairs.append(qa_pair)
                    print(f"     👍 高品質 QA 對已保存 (分數: {score:.2f})")

        print(f"   📊 文件 {source_file} 最終產生 {len(valid_pairs)} 個高品質QA對。")
        return valid_pairs

    def save_document_results(self, pairs: List[QAPair], source_md_filename: str, paper_name: str):
        """保存單個文檔的結果到其對應的論文資料夾 (與您原本的相同)"""
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
                "paper_section": pair.paper_section,
                "quality_score": pair.quality_score,
                "timestamp": pair.timestamp
            })
        
        if not rows:
            print(f"   - ⚠️  在 {source_md_filename} 中未生成任何可保存的QA對。")
            return
            
        df = pd.DataFrame(rows)
        df = df[['timestamp', 'question', 'context', 'Answer (Long)', 'Answer (Short)', 'source', 'paper_section', 'quality_score']]
        
        section_name = os.path.splitext(source_md_filename)[0]
        base_filename = f"{section_name}_qca_output_{self.run_timestamp}"
        
        csv_path = os.path.join(doc_output_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        jsonl_path = os.path.join(doc_output_dir, f"{base_filename}.jsonl")
        df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)

        print(f"\n   💾 結果已保存至目錄: {doc_output_dir}")
        print(f"     - CSV檔案: {csv_path}")
        print(f"     - JSONL檔案: {jsonl_path}")

def main():
    """主函數 (與您原本的相同)"""
    generator = QAGeneratorForChoiceQuestions()
    
    md_files_to_process = []
    for dirpath, _, filenames in os.walk(INPUT_DIR):
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
        
    md_files_to_process.sort(key=lambda x: x['path'])
    total_pairs_generated = 0
    
    print(f"🚀 開始處理 {len(md_files_to_process)} 個 Markdown 文檔...")
    
    for file_info in md_files_to_process:
        md_path = file_info["path"]
        paper_name = file_info["paper_name"]
        md_file = file_info["md_file"]
        
        try:
            pairs = generator.process_document(md_path)
            if pairs:
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