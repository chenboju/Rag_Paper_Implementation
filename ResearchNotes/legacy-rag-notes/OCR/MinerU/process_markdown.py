# import os
# import base64
# import re
# from pathlib import Path
# import time
# # 假設您已經安裝了 ollama 函式庫
# # pip install ollama
# import ollama

# def get_image_as_base64(image_path):
#     """將本地圖片文件轉換為 Base64 編碼的字符串"""
#     try:
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
#     except FileNotFoundError:
#         print(f"      錯誤: 圖片文件未找到 {image_path}")
#         return None
#     except Exception as e:
#         print(f"      讀取圖片時發生錯誤 {image_path}: {e}")
#         return None

# def call_ollama_vision_api_for_paragraph(base64_image, context_reference, caption_full):
#     """
#     調用本地 Ollama Vision API，指令其生成一段用於取代圖片的描述性正文。
#     """
#     if not context_reference:
#         context_reference = "未在內文中找到明確引用，請參考圖片的直接前文。"

#     prompt_text = f"""你是一位頂尖的學術論文作者與圖片分析專家。你的任務是根據提供的圖片和上下文，撰寫一段詳細、流暢、學術化的段落，用來在正文中**取代**這張圖片。這段文字本身需要完整地傳達圖片的所有核心信息。

# 分析重點：
# - **主要依據**: 「內文中引用此圖片的段落」是你理解圖片核心目的的關鍵。
# - **輔助參考**: 「圖片已有的標題/圖注」提供了圖片的簡要總結。

# 要求：
# 1.  **內容全面且專業**: 完整描述圖表中的趨勢、關鍵數據點、圖像中的主要元素及其相互關係。如果是實驗裝置圖，請描述其結構和流程。如果是數據圖，請解釋數據的意義和趨勢。
# 2.  **流暢的段落**: 生成的文字必須是一個或多個完整的、語法通順的段落，可以直接嵌入論文正文。
# 3.  **絕對禁止**:
#     - 不要使用 "這張圖片展示了..." 或 "根據上下文..." 這樣的引導語。
#     - 不要重複圖片的標題 (如 Fig. 1)。
#     - 不要輸出任何 Markdown 語法 (如 `**` 或 `![]()`)。

# ---
# 上下文信息：
# 1.  **內文中引用此圖片的段落 (最重要)**:
#     "{context_reference}"

# 2.  **圖片已有的標題/圖注 (供參考)**:
#     "{caption_full if caption_full else '無'}"
# ---

# 請直接開始撰寫可以取代圖片的描述性段落：
# """
#     try:
#         # 注意：請將 'gpt-oss:20b' 替換為您本地正在運行的實際模型名稱
#         response = ollama.chat(
#             model='gpt-oss:20b',
#             messages=[
#                 {
#                     'role': 'user',
#                     'content': prompt_text,
#                     'images': [base64_image]
#                 }
#             ]
#         )
#         description = response['message']['content'].strip()
#         return description
#     except Exception as e:
#         print(f"      調用 Ollama API 時發生錯誤: {e}")
#         return None

# def process_single_markdown_file(file_path, output_dir, force_update=False):
#     """
#     處理單個Markdown文件，並將結果保存到指定的輸出目錄以及原始目錄。
#     """
#     print(f"  開始處理文件: {file_path.name}")
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except Exception as e:
#         print(f"    讀取文件 {file_path.name} 時出錯: {str(e)}")
#         return

#     markdown_dir = file_path.parent
#     modified_content = content

#     pattern = re.compile(
#         r"(!\[.*?\]\(([^)]+)\))\s*"
#         r"(\n\s*(?P<fig_id>(?:Fig|Figure|Table)\s*[\d\.]+.*?))?"
#         r"(\n\s*\*\*AI 生成的詳細說明:\*\*.*?)?(\n\n|$)",
#         re.DOTALL
#     )

#     matches = list(pattern.finditer(content))
#     if not matches:
#         print("    未在文件中找到符合條件的圖片標籤。")
#         return

#     for match in reversed(matches):
#         image_path_md = match.group(2)
#         caption_line = match.group(3).strip() if match.group(3) else ""
#         figure_identifier = match.group('fig_id')
#         existing_ai_desc_block = match.group(5)

#         if existing_ai_desc_block and not force_update:
#             print(f"    圖片: {image_path_md} 已有描述，跳過。")
#             continue

#         print(f"    找到圖片: {image_path_md}")
#         actual_image_path = (markdown_dir / image_path_md).resolve()

#         context_from_reference = ""
#         if figure_identifier:
#             print(f"      找到標識: '{figure_identifier}', 搜尋引用...")
#             ref_pattern = re.compile(r'[\s\(\[,]' + re.escape(figure_identifier.strip()) + r'[\s\)\]\.,]', re.IGNORECASE)
#             ref_match = ref_pattern.search(content)
#             if ref_match:
#                 ref_pos = ref_match.start()
#                 para_start = content.rfind('\n\n', 0, ref_pos) + 2
#                 para_end = content.find('\n\n', ref_pos)
#                 if para_end == -1: para_end = len(content)
#                 context_from_reference = content[para_start:para_end].strip().replace("\n", " ")
#                 print(f"      找到引用段落: '{context_from_reference[:80]}...'")
#             else:
#                 print(f"      警告: 未找到對 '{figure_identifier}' 的引用。")

#         base64_image = get_image_as_base64(str(actual_image_path))

#         if base64_image:
#             print("      生成描述中...")
#             new_desc = call_ollama_vision_api_for_paragraph(
#                 base64_image, context_from_reference, caption_line
#             )

#             if new_desc:
#                 new_block = f"{new_desc.strip()}\n\n{caption_line}\n\n"
#                 start, end = match.start(), match.end()
#                 modified_content = modified_content[:start] + new_block + modified_content[end:]
#                 print(f"      成功替換圖片為文字描述。")
#             else:
#                 print("      無法生成描述，保留原樣。")
#         else:
#             print("      無法加載圖片，保留原樣。")

#     if content != modified_content:
#         # 建立新的檔名
#         output_filename = f"{file_path.stem}_image_text.md"

#         # --- [修改後邏輯] ---
#         # 定義兩個儲存路徑
#         central_output_path = output_dir / output_filename
#         local_output_path = file_path.with_name(output_filename) # 使用 with_name 可以在原始目錄下生成新檔名

#         # 儲存到中央目錄
#         try:
#             with open(central_output_path, 'w', encoding='utf-8') as f:
#                 f.write(modified_content)
#             print(f"  已處理並保存到中央目錄: {central_output_path}")
#         except Exception as e:
#             print(f"    寫入文件到 {central_output_path} 時出錯: {str(e)}")

#         # 儲存到原始 auto 資料夾
#         try:
#             with open(local_output_path, 'w', encoding='utf-8') as f:
#                 f.write(modified_content)
#             print(f"  同時也保存一份在原始目錄: {local_output_path}")
#         except Exception as e:
#             print(f"    寫入文件到 {local_output_path} 時出錯: {str(e)}")
#         # --- [修改結束] ---

#     else:
#         print(f"  文件無需修改: {file_path.name}")


# def find_markdown_files_in_auto(directory_path):
#     """在指定目錄下尋找所有 'auto' 子目錄中的 Markdown 文件"""
#     path = Path(directory_path)
#     if not path.is_dir():
#         print(f"錯誤: 提供的路徑 '{directory_path}' 不是一個有效的目錄。")
#         return []
    
#     markdown_files = []
#     for auto_dir in path.rglob('auto'):
#         if auto_dir.is_dir():
#             markdown_files.extend(list(auto_dir.glob('*.md')))
#             markdown_files.extend(list(auto_dir.glob('*.markdown')))
    
#     return list(set(markdown_files))

# def main():
#     # --- [請在這裡配置] ---
#     # 1. 設置包含所有論文資料夾的根目錄
#     target_directory = "E:\MinerU_Linux_10000_0523"
    
#     # 2. 設置存放所有處理後 .md 文件的【中央】輸出資料夾
#     output_directory = "E:\processed_markdowns_image_text_0915"
    
#     # 3. 是否強制更新已有描述的圖片？
#     force_update = True
#     # --- [配置結束] ---

#     target_path = Path(target_directory).resolve()
#     output_path = Path(output_directory).resolve()

#     print(f"中央輸出目錄設定為: {output_path}")
#     output_path.mkdir(parents=True, exist_ok=True)

#     print(f"開始掃描目錄: {target_path}")
#     if not target_path.exists():
#         print(f"錯誤: 目標路徑 '{target_path}' 不存在。")
#         return

#     markdown_files = find_markdown_files_in_auto(target_path)
#     if not markdown_files:
#         print("在任何 'auto' 子目錄中都未找到 Markdown 文件。")
#         return

#     print(f"找到 {len(markdown_files)} 個 Markdown 文件待處理。")
#     print("-" * 50)
    
#     for md_file in markdown_files:
#         process_single_markdown_file(md_file,
#                                      output_dir=output_path,
#                                      force_update=force_update)
#         print("-" * 50)

#     print("\n所有處理完成。")

# if __name__ == "__main__":
#     main()

import os
import base64
import re
from pathlib import Path
import time

# 假設您已經安裝了 ollama 函式庫
# pip install ollama
import ollama

def get_image_as_base64(image_path):
    """將本地圖片文件轉換為 Base64 編碼的字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"      錯誤: 圖片文件未找到 {image_path}")
        return None
    except Exception as e:
        print(f"      讀取圖片時發生錯誤 {image_path}: {e}")
        return None

def call_ollama_vision_api_for_paragraph(base64_image, context_reference, caption_full):
    """
    調用本地 Ollama Vision API，指令其生成一段用於取代圖片的描述性正文。
    """
    if not context_reference:
        context_reference = "未在內文中找到明確引用，請參考圖片的直接前文。"

    prompt_text = f"""你是一位頂尖的學術論文作者與圖片分析專家。你的任務是根據提供的圖片和上下文，撰寫一段詳細、流暢、學術化的段落，用來在正文中**取代**這張圖片。這段文字本身需要完整地傳達圖片的所有核心信息。

分析重點：
- **主要依據**: 「內文中引用此圖片的段落」是你理解圖片核心目的的關鍵。
- **輔助參考**: 「圖片已有的標題/圖注」提供了圖片的簡要總結。

要求：
1.  **內容全面且專業**: 完整描述圖表中的趨勢、關鍵數據點、圖像中的主要元素及其相互關係。如果是實驗裝置圖，請描述其結構和流程。如果是數據圖，請解釋數據的意義和趨勢。
2.  **流暢的段落**: 生成的文字必須是一個或多個完整的、語法通順的段落，可以直接嵌入論文正文。
3.  **絕對禁止**:
    - 不要使用 "這張圖片展示了..." 或 "根據上下文..." 這樣的引導語。
    - 不要重複圖片的標題 (如 Fig. 1)。
    - 不要輸出任何 Markdown 語法 (如 `**` 或 `![]()`)。

---
上下文信息：
1.  **內文中引用此圖片的段落 (最重要)**:
    "{context_reference}"

2.  **圖片已有的標題/圖注 (供參考)**:
    "{caption_full if caption_full else '無'}"
---

請直接開始撰寫可以取代圖片的描述性段落：
"""
    try:
        response = ollama.chat(
            model='gpt-oss:20b', # 請確認您的模型名稱
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [base64_image]
                }
            ]
        )
        description = response['message']['content'].strip()
        return description
    except Exception as e:
        print(f"      調用 Ollama API 時發生錯誤: {e}")
        return None

def process_single_markdown_file(file_path, output_dir):
    """
    處理單個Markdown文件，並將結果保存到指定的輸出目錄以及原始目錄。
    """
    print(f"  >> 開始處理文件: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"    讀取文件 {file_path.name} 時出錯: {str(e)}")
        return

    markdown_dir = file_path.parent
    modified_content = content
    has_changes = False

    pattern = re.compile(
        r"(!\[.*?\]\(([^)]+)\))\s*"
        r"(\n\s*(?P<fig_id>(?:Fig|Figure|Table)\s*[\d\.]+)\s*.*?)?"
        r"(\n\s*\*\*AI 生成的詳細說明:\*\*.*?)?(\n\n|$)",
        re.DOTALL
    )

    matches = list(pattern.finditer(content))
    if not matches:
        print("    未在文件中找到符合條件的圖片標籤。")
        return

    for match in reversed(matches):
        full_image_block_text = match.group(0)
        image_path_md = match.group(2)
        caption_line = match.group(3).strip() if match.group(3) else ""
        figure_identifier = match.group('fig_id')

        print(f"    找到圖片: {image_path_md}")
        actual_image_path = (markdown_dir / image_path_md).resolve()

        context_from_reference = ""
        if figure_identifier:
            print(f"      找到標識: '{figure_identifier}', 搜尋引用...")
            try:
                ref_pattern = re.compile(r'[\s\(\[,]' + re.escape(figure_identifier) + r'[\s\)\]\.,]', re.IGNORECASE)
                ref_matches = list(ref_pattern.finditer(content))
                
                for ref_match in ref_matches:
                    if match.start() <= ref_match.start() < match.end():
                        continue
                    
                    ref_pos = ref_match.start()
                    para_start = content.rfind('\n\n', 0, ref_pos) + 2
                    para_end = content.find('\n\n', ref_pos)
                    if para_end == -1: para_end = len(content)
                    
                    context_from_reference = content[para_start:para_end].strip().replace("\n", " ")
                    print(f"      找到引用段落: '{context_from_reference[:100]}...'")
                    break
                
                if not context_from_reference:
                    print(f"      警告: 未在正文中找到對 '{figure_identifier}' 的有效引用。")

            except Exception as e:
                print(f"      搜尋引用時發生錯誤: {e}")

        base64_image = get_image_as_base64(str(actual_image_path))

        if base64_image:
            print("      生成描述中...")
            new_desc = call_ollama_vision_api_for_paragraph(
                base64_image, context_from_reference, caption_line
            )

            if new_desc:
                has_changes = True
                new_block = f"{new_desc.strip()}\n\n{caption_line}\n\n"
                start, end = match.start(), match.end()
                modified_content = modified_content[:start] + new_block + modified_content[end:]
                print(f"      成功替換圖片為文字描述。")
            else:
                print("      無法生成描述，保留原樣。")
        else:
            print("      無法加載圖片，保留原樣。")

    if has_changes:
        output_filename = f"{file_path.stem}_image_text.md"
        central_output_path = output_dir / output_filename
        local_output_path = file_path.with_name(output_filename)

        try:
            with open(central_output_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"  已處理並保存到中央目錄: {central_output_path}")

            with open(local_output_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"  同時也保存一份在原始目錄: {local_output_path}")
        except Exception as e:
            print(f"    寫入文件時出錯: {str(e)}")
    else:
        print(f"  文件無需修改或未成功生成描述: {file_path.name}")

def find_markdown_files_in_auto(directory_path):
    """在指定目錄下尋找所有 'auto' 子目錄中的 Markdown 文件"""
    path = Path(directory_path)
    if not path.is_dir():
        print(f"錯誤: 提供的路徑 '{directory_path}' 不是一個有效的目錄。")
        return []
    
    markdown_files = []
    for auto_dir in path.rglob('auto'):
        if auto_dir.is_dir():
            md_in_dir = [f for f in auto_dir.glob('*.md') if '_image_text' not in f.name]
            markdown_files.extend(md_in_dir)
            
    return sorted(list(set(markdown_files)))

def main():
    # --- [請在這裡配置] ---
    target_directory = "E:/MinerU_Linux_10000_0523"
    output_directory = "E:/processed_markdowns_image_text_0915"
    force_update = False
    interactive_pause = False
    # --- [配置結束] ---

    target_path = Path(target_directory).resolve()
    output_path = Path(output_directory).resolve()

    print(f"中央輸出目錄設定為: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"開始掃描目錄: {target_path}")
    if not target_path.exists():
        print(f"錯誤: 目標路徑 '{target_path}' 不存在。")
        return

    markdown_files = find_markdown_files_in_auto(target_path)
    if not markdown_files:
        print("在任何 'auto' 子目錄中都未找到原始 Markdown 文件。")
        return

    total_files = len(markdown_files)
    processed_count = 0
    skipped_count = 0
    print(f"總共找到 {total_files} 個原始 Markdown 文件待處理。")
    print("-" * 50)
    
    for i, md_file in enumerate(markdown_files):
        print(f"進度: [{i+1}/{total_files}] - 正在檢查: {md_file.name}")

        # --- [全新優先檢查邏輯] ---
        output_filename = f"{md_file.stem}_image_text.md"
        central_output_path = output_path / output_filename

        print(f"  以中央目錄為優先檢查標準: {central_output_path.parent.name}")

        central_exists = False
        if not central_output_path.exists():
            print(f"    [✗] 未找到檔案。")
        elif central_output_path.stat().st_size == 0:
            print(f"    [✗] 找到的檔案是空的，將重新處理。")
        else:
            central_exists = True
            print(f"    [✓] 找到有效檔案: {central_output_path.name}")
            
        if central_exists and not force_update:
            print(f"  --> 結論：跳過此檔案。")
            skipped_count += 1
            print("-" * 50)
            continue
        
        if force_update:
            print(f"  --> 結論：'force_update' 為 True，將強制重新處理。")
        else:
            print(f"  --> 結論：準備開始處理。")
        # --- [檢查邏輯結束] ---

        process_single_markdown_file(md_file, output_dir=output_path)
        processed_count += 1
        
        if interactive_pause and i < total_files - 1:
            user_input = input("\n按下 Enter 繼續處理下一個檔案，或輸入 'q' 結束程式: ")
            if user_input.lower() == 'q':
                print("使用者選擇結束程式。")
                break
        
        print("-" * 50)

    print("\n所有處理完成。")
    print(f"總結：處理了 {processed_count} 個檔案，跳過了 {skipped_count} 個已處理的檔案。")

if __name__ == "__main__":
    main()