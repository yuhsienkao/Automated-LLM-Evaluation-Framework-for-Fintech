import pandas as pd
import asyncio
from openai import AsyncOpenAI
import time
import os

# --- 設定 ---
SOURCE_EXCEL_FILE = 'llm_pk_results_final.xlsx'
SOURCE_SHEET_NAME = '專家講中文'
API_KEY_SHEET_NAME = 'api_key'
OUTPUT_EXCEL_FILE = 'model_evaluation_results.xlsx'

# 要附加到每個儲存格內容後的評分規則
EVALUATION_PROMPT_SUFFIX = """
{
指令：
角色： 你是一位頂尖的財經分析師。
請將上述內容根據以下規則評分 :
詳細評分標準：
1. 內容忠實度 (Content Fidelity) - 10%
滿分標準 (10%): 完全且僅採用Prompt提供的數據與事實進行分析。所有關鍵數據均被準確無誤地引用，無任何遺漏。
部分瑕疵 (5-9%): 基本遵循Prompt內容，但出現少量非關鍵資訊的遺漏，或對數據的詮釋稍有偏離。
嚴重瑕疵 (0-4%): 包含任何Prompt以外的臆測、幻想或外部資訊；或遺漏了核心的關鍵數據。
2. 財經專業度 (Financial Proficiency) - 10%
滿分標準 (10%): 對所有財經數據、名詞的理解完全正確，並能在行文中以深入淺出、易於理解的方式加以運用或解釋。
部分瑕疵 (5-9%): 對大部分財經名詞理解正確，但對少數複雜概念的運用或解釋不夠精準。
嚴重瑕疵 (0-4%): 明顯誤解或誤用關鍵財經數據或名詞，導致分析失焦或產生誤導。
3. 邏輯結構 (Logical Structure) - 40%
滿分標準 (31-40%): 全文擁有明確的核心論點。段落之間銜接流暢，論據能有效支撐論點，形成一個有說服力且結構嚴謹的整體。
部分瑕疵 (21-30%): 核心論點基本清晰，但部分段落間的過渡稍嫌生硬，或某些論據與論點的關聯性較弱。
嚴重瑕疵 (0-20%): 缺乏明確的核心論點，內容組織混亂，段落之間缺乏關聯性，讀者難以跟隨其論述脈絡。
4. 語文表達力 (Linguistic Expression) - 40%
滿分標準 (31-40%): 使用專業、精準且流暢的中文書寫。文法正確，用詞考究，整體表達清晰自然，具備高度可讀性。
部分瑕疵 (21-30%): 語意表達大致清晰，但存在部分語句略顯冗長或繞口，或有少量語法瑕疵。
嚴重瑕疵 (0-20%): 文句不通順，存在多處語法錯誤或用詞不當，導致語意模糊，嚴重影響閱讀理解。
}
"""

# OpenRouter 上的模型識別符
MODELS = {
    "gpt-4o": "gpt-4o",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "gemini-1.5-pro": "google/gemini-1.5-pro"
}

# --- 函式定義 ---

def load_data_and_keys():
    """從 Excel 檔案載入 prompts 和 API keys"""
    try:
        # 讀取 prompts
        df_prompts = pd.read_excel(SOURCE_EXCEL_FILE, sheet_name=SOURCE_SHEET_NAME, header=None)
        prompts = df_prompts[0].dropna().tolist()
        
        # 讀取 API keys
        df_keys = pd.read_excel(SOURCE_EXCEL_FILE, sheet_name=API_KEY_SHEET_NAME, header=None, index_col=0)
        api_keys = df_keys[1].to_dict()
        
        # 檢查 keys 是否存在
        if 'openai_api_key' not in api_keys or 'openrouter_api_key' not in api_keys:
            raise ValueError("Excel 的 'api_key' sheet 中缺少 openai_api_key 或 openrouter_api_key")
            
        print(f"成功載入 {len(prompts)} 個 prompts 和 API keys。")
        return prompts, api_keys
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{SOURCE_EXCEL_FILE}'。請確認檔案名稱和路徑是否正確。")
        return None, None
    except ValueError as e:
        print(f"錯誤: {e}")
        return None, None
    except Exception as e:
        print(f"讀取 Excel 檔案時發生未知錯誤: {e}")
        return None, None

async def get_model_response(client: AsyncOpenAI, model: str, full_prompt: str, original_prompt: str, max_retries=3):
    """非同步地從指定模型獲取回應"""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1024,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"模型 {model} 處理 '{original_prompt[:20]}...' 時發生錯誤 (第 {attempt + 1} 次嘗試): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt) # 指數退讓
            else:
                return f"!!--模型處理失敗--!! 錯誤訊息: {e}"

async def main():
    """主執行函式"""
    start_time = time.time()
    
    # 1. 載入資料
    original_prompts, api_keys = load_data_and_keys()
    if original_prompts is None:
        return

    # 2. 建立 API clients
    openai_client = AsyncOpenAI(api_key=api_keys['openai_api_key'])
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_keys['openrouter_api_key'],
    )

    # 3. 準備所有任務
    tasks = []
    full_prompts = [f"{p}{EVALUATION_PROMPT_SUFFIX}" for p in original_prompts]
    
    print("正在準備 API 請求...")
    for i, (original_prompt, full_prompt) in enumerate(zip(original_prompts, full_prompts)):
        print(f"  - 準備第 {i+1}/{len(original_prompts)} 筆資料...")
        # GPT-4o 任務
        tasks.append(get_model_response(openai_client, MODELS["gpt-4o"], full_prompt, original_prompt))
        # Claude 3.5 Sonnet 任務
        tasks.append(get_model_response(openrouter_client, MODELS["claude-3.5-sonnet"], full_prompt, original_prompt))
        # Gemini 1.5 Pro 任務
        tasks.append(get_model_response(openrouter_client, MODELS["gemini-1.5-pro"], full_prompt, original_prompt))

    # 4. 平行執行所有 API 請求
    print(f"\n正在發送 {len(tasks)} 個請求至 AI 模型，請稍候...")
    results = await asyncio.gather(*tasks)
    print("所有模型回應已接收完畢。")

    # 5. 整理並儲存結果
    output_data = []
    num_models = len(MODELS)
    for i in range(len(original_prompts)):
        start_index = i * num_models
        row = {
            '原始內容': original_prompts[i],
            'GPT-4o 回應': results[start_index],
            'Claude-3.5-Sonnet 回應': results[start_index + 1],
            'Gemini-1.5-Pro 回應': results[start_index + 2]
        }
        output_data.append(row)

    df_output = pd.DataFrame(output_data)
    
    try:
        df_output.to_excel(OUTPUT_EXCEL_FILE, index=False, engine='openpyxl')
        print(f"\n成功！結果已儲存至 '{OUTPUT_EXCEL_FILE}'。")
    except Exception as e:
        print(f"儲存 Excel 檔案時發生錯誤: {e}")
        
    end_time = time.time()
    print(f"總執行時間: {end_time - start_time:.2f} 秒。")

# --- 執行主程式 ---
if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())