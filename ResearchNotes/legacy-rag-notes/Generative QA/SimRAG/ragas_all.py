import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate, run_config
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from datetime import datetime
import glob

# --- 1. 設定參數 ---

# --- vvv 修改處 vvv ---
# 指定包含所有論文資料夾的根目錄
BASE_INPUT_DIR = './output_0909' 

# 設定要儲存所有評估結果的根目錄
BASE_OUTPUT_DIR = 'ragas_evaluation_results_0911_all' 
# --- ^^^ 修改處 ^^^ ---

# 分別設定評估用的 LLM 和專門的嵌入模型
EVALUATION_LLM = "gpt-oss:20b"
EMBEDDING_MODEL = "embeddinggemma"


# --- 2. 執行評估 ---
def main():
    """主執行函數"""
    # --- vvv 修改處：使用遞迴搜尋找到所有子資料夾中的 CSV 檔案 ---
    search_path = os.path.join(BASE_INPUT_DIR, '**', '*.csv')
    csv_files = glob.glob(search_path, recursive=True)
    # --- ^^^ 修改處 ^^^ ---

    if not csv_files:
        print(f"❌ 錯誤：在資料夾 '{BASE_INPUT_DIR}' 及其子資料夾中找不到任何 CSV 檔案。")
        return

    print(f"🔍 在 '{BASE_INPUT_DIR}' 中找到 {len(csv_files)} 個 CSV 檔案，將逐一進行評估。")

    print("\n🔧 正在初始化評估模型...")
    print(f"   - 評估 LLM: {EVALUATION_LLM}")
    print(f"   - 嵌入模型: {EMBEDDING_MODEL}")
    try:
        langchain_llm = ChatOllama(model=EVALUATION_LLM, timeout=600) # 增加 timeout 時間
        ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        metrics_to_evaluate = [
            Faithfulness(), ContextRecall(), ContextPrecision(), AnswerRelevancy()
        ]
        config = run_config.RunConfig(max_workers=4) # 可以適當增加 workers 數量以加速
        print("✅ 模型初始化完成。")
    except Exception as e:
        print(f"❌ 模型初始化失敗：{e}")
        print("   - 請確認 OLLAMA 服務是否已啟動，且模型 '{EVALUATION_LLM}' 和 '{EMBEDDING_MODEL}' 已正確安裝。")
        return


    for input_csv_path in csv_files:
        try:
            print(f"\n{'='*80}")
            # 顯示檔案的相對路徑，更清晰
            relative_path = os.path.relpath(input_csv_path, BASE_INPUT_DIR)
            print(f"🚀 開始處理檔案: {relative_path}")
            print(f"{'='*80}")

            df = pd.read_csv(input_csv_path)
            
            # 篩選掉沒有答案的行，避免後續處理出錯
            df.dropna(subset=['Answer (Long)', 'Answer (Short)', 'context'], inplace=True)
            if df.empty:
                print("   - 篩選後無有效資料，跳過此檔案。")
                continue

            print(f"   - 資料集共 {len(df)} 筆，將全部進行評估。")

            print("   - 正在準備 Ragas 所需的資料格式...")
            df = df.rename(columns={
                'Answer (Long)': 'answer',
                'Answer (Short)': 'ground_truth',
                'Question': 'question' # 確保問題欄位也被正確命名
            })
            df['contexts'] = df['context'].apply(lambda x: [str(x)] if pd.notna(x) else [])
            
            required_columns = ['question', 'answer', 'contexts', 'ground_truth']
            for col in required_columns:
                if col not in df.columns:
                    raise KeyError(f"錯誤：CSV 檔案中缺少必要的欄位 '{col}'。")
            
            dataset = Dataset.from_pandas(df[required_columns])
            print(f"✅ 資料集已成功載入並準備完成。")

            print("   - (本地模型運算較慢，請耐心等候...)")
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_evaluate,
                llm=langchain_llm,
                embeddings=ollama_embeddings,
                run_config=config,
                raise_exceptions=False # 評估單筆失敗時不中斷，繼續執行
            )
            print("✅ 評估完成！")

            result_df = result.to_pandas()
            print("\n📝 評估結果摘要 (各指標平均分數)：")
            # 計算平均分數時，只看分數欄位
            score_columns = ['faithfulness', 'context_recall', 'context_precision', 'answer_relevancy']
            mean_scores = result_df[score_columns].mean()
            print(mean_scores)

            # --- vvv 修改處：建立對應的輸出子資料夾並儲存檔案 ---
            # 取得輸入檔案相對於基礎目錄的路徑 (e.g., 3D-ink-extrusion.../1_ABSTRACT_...csv)
            relative_path = os.path.relpath(input_csv_path, BASE_INPUT_DIR)
            
            # 建立在輸出目錄中對應的子資料夾路徑 (e.g., ragas_results/3D-ink-extrusion...)
            output_subdir = os.path.join(BASE_OUTPUT_DIR, os.path.dirname(relative_path))
            os.makedirs(output_subdir, exist_ok=True)
            
            # 取得原始檔名 (不含副檔名)
            input_filename = os.path.basename(input_csv_path)
            base_name = os.path.splitext(input_filename)[0]
            
            # 組合新的輸出檔名和路徑
            output_csv_filename = f"ragas_eval_{base_name}.csv" 
            output_csv_path = os.path.join(output_subdir, output_csv_filename)
            result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 完整的 CSV 評估結果已儲存至: '{output_csv_path}'")

            columns_to_drop = ['context', 'contexts']
            existing_columns_to_drop = [col for col in columns_to_drop if col in result_df.columns]
            df_for_json = result_df.drop(columns=existing_columns_to_drop)
            
            output_json_filename = f"ragas_eval_{base_name}.json" 
            output_json_path = os.path.join(output_subdir, output_json_filename)
            # --- ^^^ 修改處 ^^^ ---
            
            df_for_json.to_json(
                output_json_path, orient='records', lines=False, indent=4, force_ascii=False
            )
            print(f"💾 不含上下文的 JSON 評估結果已儲存至: '{output_json_path}'")

        except FileNotFoundError:
            print(f"❌ 錯誤：找不到檔案 '{input_csv_path}'。")
        except KeyError as e:
            print(f"❌ 處理檔案 '{os.path.basename(input_csv_path)}' 時發生欄位錯誤 - {e}")
        except Exception as e:
            print(f"❌ 處理檔案 '{os.path.basename(input_csv_path)}' 時發生未預期的錯誤：{e}")
            import traceback
            traceback.print_exc() # 印出詳細的錯誤堆疊
            print("   - 請檢查 OLLAMA 服務及模型是否正常，或 CSV 檔案內容格式是否正確。")
            
if __name__ == "__main__":
    main()