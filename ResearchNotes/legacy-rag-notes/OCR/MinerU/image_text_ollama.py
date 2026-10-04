import os
import base64
import re
from pathlib import Path
import time
import ollama

def get_image_as_base64(image_path):
    """將本地圖片文件轉換為 Base64 編碼的字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"    錯誤: 圖片文件未找到 {image_path}")
        return None
    except Exception as e:
        print(f"    讀取圖片時發生錯誤 {image_path}: {e}")
        return None

def call_ollama_vision_api_for_paragraph(base64_image, context_immediate, context_reference, caption_full):
    """
    調用本地 Ollama Vision API，指令其生成一段用於取代圖片的描述性正文。
    """
    if not context_reference:
        context_reference = "未在內文中找到明確引用，請參考圖片的直接前文。"

    # --- [核心修改] 更新 Prompt，指令 AI 撰寫替代圖片的段落 ---
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
            model='qwen3:30b-a3b-instruct-2507-q4_K_M',
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
        print(f"    調用 Ollama API 時發生錯誤: {e}")
        return None

def process_single_markdown_file(file_path, force_add_detailed_desc=False, create_new_file=True):
    """
    處理單個Markdown文件，並按照新的格式要求重組內容。
    """
    print(f"  開始處理文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"    讀取文件 {file_path} 時出錯: {str(e)}")
        return

    markdown_dir = Path(file_path).parent
    modified_in_this_file = False
    
    image_pattern = re.compile(
        r"(!\[(.*?)\]\((.*?)\))"  # Group 1, 2, 3: 完整的圖片標籤, alt text, 圖片路徑
        r"(\s*\n\s*(?P<fig_id>(Fig\.|Figure|Table)\s*[\d\.]+)\s*(.*?)\n)?" # Group 4, 5, 6: 標題行, 標識符, 標題文本
        r"(?P<ai_desc_block>\s*\n\s*\*\*AI 生成的詳細說明:\*\*.+?)?(\n\n|$)", # AI 區塊
        re.DOTALL
    )

    processed_content_parts = []
    last_match_end = 0
    
    for match in image_pattern.finditer(content):
        processed_content_parts.append(content[last_match_end:match.start()])
        
        full_match_text = match.group(0)
        image_path_md = match.group(3)
        caption_line = match.group(4).strip() if match.group(4) else ""
        figure_identifier = match.group('fig_id')
        existing_ai_desc_block = match.group('ai_desc_block')

        if existing_ai_desc_block and not force_add_detailed_desc:
            print(f"    圖片: {image_path_md} 已存在AI描述且未強制更新，跳過。")
            processed_content_parts.append(full_match_text)
            last_match_end = match.end()
            continue

        print(f"    找到圖片: {image_path_md}")
        context_from_reference = ""
        if figure_identifier:
            print(f"      找到圖片標識: '{figure_identifier}'，開始搜尋引用位置...")
            ref_pattern_str = r'[\s\(\[,]' + re.escape(figure_identifier) + r'[\s\)\]\.,]'
            reference_pattern = re.compile(ref_pattern_str, re.IGNORECASE)
            ref_match = reference_pattern.search(content)
            if ref_match:
                ref_pos = ref_match.start()
                para_start = content.rfind('\n\n', 0, ref_pos)
                para_start = 0 if para_start == -1 else para_start + 2
                para_end = content.find('\n\n', ref_pos)
                if para_end == -1: para_end = len(content)
                context_from_reference = content[para_start:para_end].strip().replace("\n", " ")
                print(f"      成功定位引用段落: '{context_from_reference[:100]}...'")
            else:
                print(f"      警告: 在文中未找到對 '{figure_identifier}' 的引用。")

        actual_image_path = (markdown_dir / image_path_md).resolve()
        base64_image = get_image_as_base64(str(actual_image_path))

        if base64_image:
            context_before_image = content[max(0, match.start() - 1000):match.start()].strip()
            print(f"      正在為 '{actual_image_path.name}' 生成詳細文字說明...")
            
            new_detailed_description = call_ollama_vision_api_for_paragraph(
                base64_image, context_before_image, context_from_reference, caption_line
            )

            if new_detailed_description:
                modified_in_this_file = True
                preview_desc = new_detailed_description.replace('\n', ' ')[:100]
                print(f"      LLM 生成的說明: '{preview_desc}...'")
                
                # --- [核心修改] 按照 新格式 (文字說明 -> 標題) 構建文本塊 ---
                # 這個新區塊會完全取代舊的 ![]() + Fig... + **AI...** 區塊
                new_block = f"{new_detailed_description.strip()}\n\n{caption_line}\n\n"
                processed_content_parts.append(new_block)
            else:
                print(f"      無法生成詳細說明，保留原始結構。")
                processed_content_parts.append(full_match_text)
        else:
            print(f"      無法加載圖片，保留原始結構。")
            processed_content_parts.append(full_match_text)
            
        last_match_end = match.end()

    processed_content_parts.append(content[last_match_end:])
    new_content = "".join(processed_content_parts)

    if modified_in_this_file:
        output_path = file_path
        if create_new_file:
            p = Path(file_path)
            output_path = p.with_name(f"{p.stem}_processed{p.suffix}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  已處理並保存到 {'新文件' if create_new_file else '文件'}: {output_path}")
        except Exception as e:
            print(f"    寫入文件 {output_path} 時出錯: {str(e)}")
    else:
        print(f"  文件無需修改: {file_path}")

def find_markdown_files(directory_path):
    path = Path(directory_path)
    if path.is_file() and path.suffix.lower() in ('.md', '.markdown'):
        return [path]
    elif path.is_dir():
        return list(path.rglob('auto/*.md')) + list(path.rglob('auto/*.markdown'))
    else:
        print(f"提供的路徑 '{directory_path}' 不是有效的文件或目錄。")
        return []

def main():
    target_path = r"C:\Users\chen\Desktop\GitHub\MinerU\MinerU_HEA_0909\豪哥"
    force_description_update = True
    generate_new_files = True

    print(f"開始掃描目錄: {Path(target_path).resolve()}")
    if not Path(target_path).exists():
        print(f"錯誤: 目標路徑 '{target_path}' 不存在。")
        return

    markdown_files = find_markdown_files(target_path)
    if not markdown_files:
        print("在 'auto' 子目錄中未找到任何 Markdown 文件。")
        return

    print(f"找到 {len(markdown_files)} 個 Markdown 文件待處理。")
    for md_file in markdown_files:
        process_single_markdown_file(str(md_file),
                                     force_add_detailed_desc=force_description_update,
                                     create_new_file=generate_new_files)
        print("-" * 40)
    
    print("\n所有處理完成。")

if __name__ == "__main__":
    main()