import pandas as pd
import os
import json
import time
import requests
import datetime
from tqdm import tqdm

# 設定輸出結果的資料夾名稱
output_dir = "results"
# 如果資料夾不存在，則建立它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 設定Excel檔案路徑
file_path = "LLM PK v2.xlsx"
try:
    # 讀取"比較表"工作表
    comparison_df = pd.read_excel(file_path, sheet_name="比較表")
    
    # 讀取"promot for PK"工作表
    prompts_df = pd.read_excel(file_path, sheet_name="promot for PK")
    
    # 讀取"API KEY"工作表
    api_keys_df = pd.read_excel(file_path, sheet_name="API KEY")
    
    # 定義不同解析規則的系統提示詞
    rules = {
        "專家講中文": "你是一位專業的中文專家，請用流暢、專業的中文回答以下問題。",
        "輕解析": "請進行簡明扼要的輕度解析，抓住要點並提供精確的回答。",
        "解析": "請深入解析問題，提供詳盡的分析和全面的說明。"
    }

    # 定義第一組模型的列表
    group1_models_list = [
        "GPT-4o-mini", "GPT-4.1-mini", "GPT-4.1-nano", "Claude-3.5-Haiku", "Gemini-2.5-Flash",
        "Grok 3 Mini", "DeepSeek-V3-0324", "mistral-small-2503", "Llama-4-Scout-17B-16E-Instruct",
        "Llama-4-Maverick-17B-128E-Instruct", "Phi-4", "gemma3-27b", "Llama 4 Scout (17Bx16E)",
        "Llama 4 Maverick (17Bx128E)", "Llama Guard 4 12B 128k", "Llama 3.3 70B Versatile 128k",
        "Mistral Saba 24B", "Gemma 2 9B 8k"
    ]

    # 定義第二組模型的列表
    group2_models_list = [
        "GPT-4o", "GPT-4o-mini", "GPT-4.1", "GPT-4.1-mini", "GPT-4.1-nano", "Claude-3.5-Haiku",
        "Claude-3.7-Sonnet", "Gemini-2.5-Flash", "Gemini-2.5-Pro", "Grok 3", "Grok 3 Mini",
        "DeepSeek-V3-0324", "Mistral Large 3", "Llama-4-Scout-17B-16E-Instruct", 
        "Llama-4-Maverick-17B-128E-Instruct", "mistral-small-2503", "Phi-4", "gemma3-27b",
        "Llama 4 Scout (17Bx16E)", "Llama 4 Maverick (17Bx128E)", "Llama Guard 4 12B 128k",
        "Llama 3.3 70B Versatile 128k", "DeepSeek R1 Distill Llama 70B", "Mistral Saba 24B",
        "Gemma 2 9B 8k"
    ]

    # 初始化模型列表
    group1_models = [] 
    group2_models = []  

    # 建立一個字典來儲存模型的API來源
    model_api_sources = {} 

    # 遍歷"比較表"，從中提取模型名稱和API來源
    for i in range(1, comparison_df.shape[0]):
        if pd.notna(comparison_df.iloc[i, 1]):
            model_name = str(comparison_df.iloc[i, 1]).strip()
            
            # 確保模型名稱有效且不是特定排除的名稱
            if (model_name and model_name != "nan" and 
                "microsoft financial-reports-analysis" not in model_name.lower()):
                
                api_source = ""
                if pd.notna(comparison_df.iloc[i, 2]):
                    api_source = str(comparison_df.iloc[i, 2]).strip()
                
                # 將模型名稱和API來源存入字典
                model_api_sources[model_name] = api_source

    # 根據group1_models_list建立第一組模型的詳細列表
    for model_name in group1_models_list:
        api_source = model_api_sources.get(model_name, "")  
        group1_models.append({"name": model_name, "api_source": api_source})

    # 根據group2_models_list建立第二組模型的詳細列表
    for model_name in group2_models_list:
        api_source = model_api_sources.get(model_name, "")  
        group2_models.append({"name": model_name, "api_source": api_source})

    # 設定第二組測試時使用的模型列表 (此處設定為與第一組相同)
    group2_models_for_testing = group1_models

    # 印出第一組模型的列表
    print("\n第一組模型:")
    for i, model in enumerate(group1_models):
        print(f"  {i+1}. {model['name']} ({model['api_source']})")
    
    # 印出第二組模型的列表
    print(f"\n第二組模型:")
    for i, model in enumerate(group2_models):
        print(f"  {i+1}. {model['name']} ({model['api_source']})")
    
    # 初始化API金鑰字典
    api_keys = {}
    # 從api_keys_df讀取API金鑰
    for i, row in api_keys_df.iterrows():
        if len(row) >= 2:  
            service = row.iloc[0]
            key = row.iloc[1]
            if isinstance(service, str) and isinstance(key, str):
                api_keys[service.strip().lower()] = key.strip()

    # 設定Groq的API金鑰 (請替換為您自己的金鑰)
    api_keys["groq"] = "YOUR_GROQ_API_KEY_HERE"

    # 定義必要的API提供商
    required_providers = ["openai", "openrouter", "groq"]
    # 檢查是否缺少必要的API金鑰
    missing_providers = [provider for provider in required_providers if provider not in api_keys]

    # 印出API金鑰 (隱藏部分金鑰內容以保護隱私)
    print("\nAPI Keys:")
    for service, key in api_keys.items():
        masked_key = key[:5] + "..." + key[-5:] if len(key) > 10 else key
        print(f"  - {service}: {masked_key}")

    # 如果有缺少的API金鑰，則發出警告並詢問是否繼續
    if missing_providers:
        print(f"\n警告：缺少以下API提供商的金鑰：{', '.join(missing_providers)}")
        print("程式將僅處理有對應API金鑰的模型。若要測試所有模型，請確保提供所有必要的API金鑰。")
        
        while True:
            proceed = input("\n是否仍要繼續執行程式？(y/n): ")
            if proceed.lower() in ['y', 'yes', 'n', 'no']:
                break
        
        if proceed.lower() in ['n', 'no']:
            print("程式已終止。")
            exit()
    
    # 印出找到的prompt數量
    print(f"\n找到 {len(prompts_df)} 個prompts")
    
    # 將prompts分組
    group1_prompts = prompts_df.iloc[:20]
    group2_prompts = prompts_df.iloc[20:26]
    group3_prompts = prompts_df.iloc[26:46]
    
    # 印出各組prompt的數量
    print(f"第一組 (專家講中文): {len(group1_prompts)} 個 (1-20)")
    print(f"第二組 (輕解析): {len(group2_prompts)} 個 (21-26)")
    print(f"第三組 (解析): {len(group3_prompts)} 個 (27-46)")
    
    # 計算總共需要調用API的次數
    total_calls = (len(group1_prompts) * len(group1_models) + 
                   len(group2_prompts) * len(group1_models) +  
                   len(group3_prompts) * len(group2_models)) 
    
    # 印出總調用次數和各組的細項
    print(f"\n總計調用次數: {total_calls}")
    print(f"- 第一組: {len(group1_prompts)} prompts x {len(group1_models)} models = {len(group1_prompts) * len(group1_models)}")
    print(f"- 第二組: {len(group2_prompts)} prompts x {len(group1_models)} models = {len(group2_prompts) * len(group1_models)}")
    print(f"- 第三組: {len(group3_prompts)} prompts x {len(group2_models)} models = {len(group3_prompts) * len(group2_models)}")
    
    # 詢問使用者是否確定要開始調用API
    while True:
        answer = input("\n是否要繼續實際調用API？這將消耗API額度(y/n): ")
        if answer.lower() in ['y', 'yes', 'n', 'no']:
            break

    # 如果使用者選擇否，則終止程式
    if answer.lower() in ['n', 'no']:
        print("程式已終止。")
        exit()
    
    def call_openai_api(prompt, model_name, api_key, rule=""):
        """調用OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # OpenAI模型名稱映射，將通用名稱轉換為API接受的具體名稱
        openai_model_mapping = {
            "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
            "GPT-4.1-mini": "gpt-4.1-mini-2025-04-14",
            "GPT-4.1-nano": "gpt-4.1-nano-2025-04-14",
            "GPT-4o": "gpt-4o-2024-08-06",
            "GPT-4.1": "gpt-4.1-2025-04-14",
            "GPT-4": "gpt-4"
        }
        
        # 進行模型名稱映射
        if model_name in openai_model_mapping:
            actual_model = openai_model_mapping[model_name]
            print(f"    映射模型(OpenAI): {model_name} -> {actual_model}")
        else:
            actual_model = model_name
        
        # 準備API請求的訊息內容
        messages = []
        if rule:
            messages.append({"role": "system", "content": rule})
        messages.append({"role": "user", "content": prompt})
        
        # 準備API請求的資料
        data = {
            "model": actual_model,
            "messages": messages,
            "temperature": 0.7
        }
        
        # 記錄API調用的開始和結束時間
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        
        # 處理API回應
        if response.status_code == 200:
            response_json = response.json()
            return {
                "response": response_json["choices"][0]["message"]["content"],
                "status_code": response.status_code,
                "time_taken": end_time - start_time,
                "token_usage": response_json.get("usage", {})
            }
        else:
            return {
                "response": f"Error: {response.status_code} - {response.text}",
                "status_code": response.status_code,
                "time_taken": end_time - start_time,
                "token_usage": {}
            }
    
    def call_openrouter_api(prompt, model_name, api_key, rule=""):
        """調用OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://localhost",  # OpenRouter要求提供此標頭
            "X-Title": "LLM PK"
        }
        
        # OpenRouter模型名稱映射
        openrouter_model_mapping = {
            "Claude-3.5-Haiku": "anthropic/claude-3.5-haiku",
            "Claude 3.5 Haiku": "anthropic/claude-3.5-haiku",
            "Claude-3-Haiku": "anthropic/claude-3-haiku",
            "Claude 3 Haiku": "anthropic/claude-3-haiku",
            "Claude 3 Sonnet": "anthropic/claude-3-sonnet",
            "Claude-3-Sonnet": "anthropic/claude-3-sonnet",
            "Claude-3.5-Sonnet": "anthropic/claude-3.5-sonnet",
            "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
            "Claude-3.7-Sonnet": "anthropic/claude-3.7-sonnet",
            "Claude 3.7 Sonnet": "anthropic/claude-3.7-sonnet",
            "Claude 3 Opus": "anthropic/claude-3-opus",
            "Claude-3-Opus": "anthropic/claude-3-opus",
            
            "Gemini-2.5-Flash": "google/gemini-2.5-flash-preview",
            "Gemini 2.5 Flash": "google/gemini-2.5-flash-preview",
            "Gemini-2.5-Pro": "google/gemini-2.5-pro-preview",
            "Gemini 2.5 Pro": "google/gemini-2.5-pro-preview",
            "Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
            
            "Grok 3": "x-ai/grok-3-beta",
            "Grok 3 Mini": "x-ai/grok-3-mini-beta",
            
            "DeepSeek-V3-0324": "deepseek/deepseek-chat-v3-0324",
            "DeepSeek-V3": "deepseek/deepseek-chat-v3-0324",
            
            "Mistral Large 3": "mistralai/mistral-large",
            "mistral-small-2503": "mistralai/mistral-small-3.1-24b-instruct",
            "Mistral Small 24B": "mistralai/mistral-small-3.1-24b-instruct",
            
            "Llama-4-Scout-17B-16E-Instruct": "meta-llama/llama-4-scout",
            "Llama 4 Scout (17Bx16E)": "meta-llama/llama-4-scout",
            "Llama-4-Maverick-17B-128E-Instruct": "meta-llama/llama-4-maverick",
            "Llama 4 Maverick (17Bx128E)": "meta-llama/llama-4-maverick",
            
            "Phi-4": "microsoft/phi-4",
            "gemma3-27b": "google/gemma-3-27b-it"
        }
        
        # 進行模型名稱映射
        if model_name in openrouter_model_mapping:
            actual_model = openrouter_model_mapping[model_name]
            print(f"    映射模型(OpenRouter): {model_name} -> {actual_model}")
        else:
            actual_model = model_name
        
        # 準備API請求的訊息內容
        messages = []
        if rule:
            messages.append({"role": "system", "content": rule})
        messages.append({"role": "user", "content": prompt})
        
        # 準備API請求的資料
        data = {
            "model": actual_model,
            "messages": messages,
            "temperature": 0.7
        }
        
        # 記錄API調用的開始和結束時間
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        
        # 處理API回應
        if response.status_code == 200:
            response_json = response.json()
            return {
                "response": response_json["choices"][0]["message"]["content"],
                "status_code": response.status_code,
                "time_taken": end_time - start_time,
                "token_usage": response_json.get("usage", {})
            }
        else:
            return {
                "response": f"Error: {response.status_code} - {response.text}",
                "status_code": response.status_code,
                "time_taken": end_time - start_time,
                "token_usage": {}
            }
    
    def call_groq_api(prompt, model_name, api_key, rule=""):
        """調用Groq API"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Groq模型名稱映射
        groq_model_mapping = {
            "Llama 4 Scout (17Bx16E)": "meta-llama/llama-4-scout-17b-16e-instruct",
            "Llama-4-Scout-17B-16E-Instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
            "Llama 4 Maverick (17Bx128E)": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "Llama-4-Maverick-17B-128E-Instruct": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "Llama Guard 4 12B 128k": "meta-llama/Llama-Guard-4-12B",
            "Llama 3.3 70B Versatile 128k": "llama-3.3-70b-versatile",
            
            "Mistral Saba 24B": "mistral-saba-24b", 
            
            "Gemma 2 9B 8k": "gemma2-9b-it",
            "gemma3-27b": "google/gemma-3-27b-it",
            
            "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b"
        }
        
        # 進行模型名稱映射
        if model_name in groq_model_mapping:
            actual_model = groq_model_mapping[model_name]
            print(f"    映射模型(Groq): {model_name} -> {actual_model}")
        else:
            actual_model = model_name
        
        # 準備API請求的訊息內容
        messages = []
        if rule:
            messages.append({"role": "system", "content": rule})
        messages.append({"role": "user", "content": prompt})
        
        # 準備API請求的資料
        data = {
            "model": actual_model,
            "messages": messages,
            "temperature": 0.7
        }
        
        # 記錄API調用的開始和結束時間
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        
        # 處理API回應
        if response.status_code == 200:
            response_json = response.json()
            return {
                "response": response_json["choices"][0]["message"]["content"],
                "status_code": response.status_code,
                "time_taken": end_time - start_time,
                "token_usage": response_json.get("usage", {})
            }
        else:
            return {
                "response": f"Error: {response.status_code} - {response.text}",
                "status_code": response.status_code,
                "time_taken": end_time - start_time,
                "token_usage": {}
            }
    
    def identify_api_provider(model_name, api_source):
        """根據模型名稱和API來源識別API提供商"""
        api_source_lower = api_source.lower() if api_source else ""
        model_name_lower = model_name.lower()
        
        # 首先根據API來源欄位判斷
        if "openai" in api_source_lower or "官方" in api_source_lower:
            return "openai"
        elif "openrouter" in api_source_lower:
            return "openrouter"
        elif "groq" in api_source_lower or "groqcloud" in api_source_lower:
            return "groq"
        
        # 如果API來源欄位不明確，則根據模型名稱的關鍵字來判斷
        openai_models = [
            "GPT-4o-mini", "GPT-4.1-mini", "GPT-4.1-nano", "GPT-4o", "GPT-4.1", "GPT-4"
        ]
        
        openrouter_models = [
            "Claude-3.5-Haiku", "Claude-3.7-Sonnet", "Gemini-2.5-Flash", "Gemini-2.5-Pro",
            "Grok 3", "Grok 3 Mini", "DeepSeek-V3-0324", "Mistral Large 3",
            "Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct",
            "mistral-small-2503", "Phi-4", "gemma3-27b"
        ]
        
        groq_models = [
            "Llama 4 Scout (17Bx16E)", "Llama 4 Maverick (17Bx128E)",
            "Llama Guard 4 12B 128k", "Llama 3.3 70B Versatile 128k",
            "DeepSeek R1 Distill Llama 70B", "Mistral Saba 24B", "Gemma 2 9B 8k"
        ]
        
        # 根據預先定義好的模型列表判斷
        if model_name in openai_models:
            return "openai"
        elif model_name in openrouter_models:
            return "openrouter"
        elif model_name in groq_models:
            return "groq"
        
        # 如果模型不在預定義列表中，則根據模型名稱中的關鍵字進行更廣泛的判斷
        if model_name_lower.startswith("gpt") or "gpt-4" in model_name_lower:
            return "openai"
        
        elif "claude" in model_name_lower:
            return "openrouter"  
        
        elif "gemini" in model_name_lower:
            return "openrouter"  
        
        elif "grok" in model_name_lower:
            return "openrouter"  
        
        elif "deepseek" in model_name_lower:
            if "r1 distill" in model_name_lower:
                return "groq"  
            else:
                return "openrouter"  
        
        elif "mistral" in model_name_lower:
            if "large" in model_name_lower:
                return "openrouter"  
            elif "saba" in model_name_lower:
                return "groq"  
            else:
                return "openrouter"  
        
        elif "llama" in model_name_lower:
            if "scout" in model_name_lower or "maverick" in model_name_lower:
                if "(17bx" in model_name_lower:
                    return "groq"  
                else:
                    return "openrouter"  
            else:
                return "groq"  
        
        elif "phi-4" in model_name_lower or "phi-3" in model_name_lower:
            return "openrouter"  
        
        elif "gemma" in model_name_lower:
            if "9b" in model_name_lower:
                return "groq"  
            else:
                return "openrouter"  
        
        # 如果以上規則都無法判斷，則預設使用OpenRouter
        return "openrouter"
    
    def call_llm_api(prompt, model, prompt_number, rule=""):
        """根據模型選擇對應的API"""
        model_name = model["name"]
        api_source = model["api_source"]
        
        # 識別API提供商
        api_provider = identify_api_provider(model_name, api_source)
        
        # 設定最大重試次數
        max_retries = 2
        retry_count = 0
        
        # 包含重試機制的API調用循環
        while retry_count <= max_retries:
            try:
                # 根據API提供商調用對應的函式
                if api_provider == "openai":
                    if "openai" in api_keys:
                        result = call_openai_api(prompt, model_name, api_keys["openai"], rule)
                        # 提取 token 資訊
                        token_usage = result["token_usage"]
                        prompt_tokens = token_usage.get("prompt_tokens", 0)
                        completion_tokens = token_usage.get("completion_tokens", 0)
                        
                        return {
                            "model": model_name,
                            "prompt_number": prompt_number,
                            "api_source": api_source,
                            "api_provider": api_provider,
                            "response": result["response"],
                            "status_code": result["status_code"],
                            "time_taken": result["time_taken"],
                            "token_usage": result["token_usage"],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "rule": rule    
                        }
                    else:
                        return {"error": "No OpenAI API key found"}
                elif api_provider == "openrouter":
                    if "openrouter" in api_keys:
                        result = call_openrouter_api(prompt, model_name, api_keys["openrouter"], rule)
                        # 提取 token 資訊
                        token_usage = result["token_usage"]
                        prompt_tokens = token_usage.get("prompt_tokens", 0)
                        completion_tokens = token_usage.get("completion_tokens", 0)
                        
                        return {
                            "model": model_name,
                            "prompt_number": prompt_number,
                            "api_source": api_source,
                            "api_provider": api_provider,
                            "response": result["response"],
                            "status_code": result["status_code"],
                            "time_taken": result["time_taken"],
                            "token_usage": result["token_usage"],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "rule": rule    
                        }
                    else:
                        return {"error": "No OpenRouter API key found"}
                elif api_provider == "groq":
                    if "groq" in api_keys:
                        result = call_groq_api(prompt, model_name, api_keys["groq"], rule)
                        # 提取 token 資訊
                        token_usage = result["token_usage"]
                        prompt_tokens = token_usage.get("prompt_tokens", 0)
                        completion_tokens = token_usage.get("completion_tokens", 0)
                        
                        return {
                            "model": model_name,
                            "prompt_number": prompt_number,
                            "api_source": api_source,
                            "api_provider": api_provider,
                            "response": result["response"],
                            "status_code": result["status_code"],
                            "time_taken": result["time_taken"],
                            "token_usage": result["token_usage"],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "rule": rule    
                        }
                    else:
                        return {"error": "No Groq API key found"}
                else:
                    return {
                        "model": model_name,
                        "prompt_number": prompt_number,
                        "api_source": api_source,
                        "api_provider": "unknown",
                        "response": f"Error: Unsupported API provider '{api_provider}' for model '{model_name}'",
                        "status_code": 0,
                        "time_taken": 0,
                        "token_usage": {},
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "rule": rule    
                    }
            except Exception as e:  
                # 如果達到最大重試次數，則返回錯誤訊息
                if retry_count >= max_retries:
                    return {
                        "model": model_name,
                        "prompt_number": prompt_number,
                        "api_source": api_source,
                        "api_provider": api_provider if 'api_provider' in locals() else "unknown",
                        "response": f"Error after {max_retries} retries: {str(e)}",
                        "status_code": 0,
                        "time_taken": 0,
                        "token_usage": {},
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "rule": rule    
                    }
                # 如果未達到最大重試次數，則進行重試
                else:
                    retry_count += 1
                    print(f"    錯誤，進行第 {retry_count} 次重試: {str(e)}")
                    time.sleep(2 * retry_count)
    
    # 獲取當前時間戳，用於命名檔案
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 設定中間結果和最終結果的檔案路徑
    intermediate_result_file = os.path.join(output_dir, f"llm_pk_results_{timestamp}_intermediate.json")
    excel_file = os.path.join(output_dir, f"llm_pk_results_{timestamp}_live.xlsx")
    final_excel_file = os.path.join(output_dir, f"llm_pk_results_{timestamp}_final.xlsx")

    # 初始化已處理操作的計數器
    processed_count = 0

    def save_results_realtime(results_data, json_file, excel_file, force=False):
        """實時儲存結果到JSON和Excel"""
        global processed_count
        
        processed_count += 1
        
        # 每處理5次操作或強制儲存時，執行儲存動作
        if force or processed_count % 5 == 0:
            # 儲存到JSON檔案
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            # 儲存到Excel檔案
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
                # 遍歷每個結果組
                for group, group_results in results_data.items():
                    if group_results:
                        df = pd.DataFrame(group_results)
                        # 設定Excel工作表中欄位的順序
                        if 'rule' in df.columns and len(df) > 0:
                            column_order = ['model', 'prompt_number', 'rule', 'response', 'status_code', 
                                            'time_taken', 'api_provider', 'api_source', 'prompt_tokens', 'completion_tokens', 'token_usage', 'timestamp']
                            for col in df.columns:
                                if col not in column_order:
                                    column_order.append(col)
                            visible_columns = [col for col in column_order if col in df.columns]
                            df = df[visible_columns]
                        df.to_excel(writer, sheet_name=group, index=False)
                
                # 生成並儲存模型統計摘要
                models_summary = generate_model_summary(results_data)
                for group, models in models_summary.items():
                    if models:
                        rows = []
                        for model_name, stats in models.items():
                            total_prompts = stats["成功"] + stats["失敗"]
                            completion_rate = (stats["成功"] / total_prompts * 100) if total_prompts > 0 else 0
                            
                            avg_api_time = (stats["API時間總和"] / stats["成功調用數"]) if stats["成功調用數"] > 0 else 0
                            avg_process_time = (stats["處理時間總和"] / stats["成功調用數"]) if stats["成功調用數"] > 0 else 0
                            
                            rows.append({
                                "模型名稱": model_name,
                                "成功呼叫數": stats["成功"],
                                "失敗呼叫數": stats["失敗"],
                                "總呼叫數": total_prompts,
                                "完成率(%)": round(completion_rate, 2),
                                "平均API時間(秒)": round(avg_api_time, 2),
                                "平均總處理時間(秒)": round(avg_process_time, 2),
                                "測試的提示詞數量": len(stats["提示詞"]),
                                "測試的提示詞編號": ", ".join([str(p) for p in sorted(stats["提示詞"])]),
                                "回答規則": group
                            })
                        
                        if rows:
                            summary_df = pd.DataFrame(rows)
                            summary_df = summary_df.sort_values(by=["完成率(%)"], ascending=False)
                            sheet_name = f"{group}_模型統計"[:31] 
                            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 生成並儲存提示詞統計摘要
                prompts_summary = generate_prompt_summary(results_data)
                for group, prompts in prompts_summary.items():
                    if prompts:
                        rows = []
                        for prompt_number, stats in prompts.items():
                            total_models = stats["成功"] + stats["失敗"]
                            completion_rate = (stats["成功"] / total_models * 100) if total_models > 0 else 0
                            rows.append({
                                "提示詞編號": prompt_number,
                                "成功調用數": stats["成功"],
                                "失敗調用數": stats["失敗"],
                                "總調用數": total_models,
                                "完成率(%)": round(completion_rate, 2),
                                "測試的模型數量": len(stats["模型"]),
                                "回答規則": group
                            })
                        
                        if rows:
                            summary_df = pd.DataFrame(rows)
                            summary_df = summary_df.sort_values(by=["提示詞編號"])
                            sheet_name = f"{group}_提示詞統計"[:31] 
                            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 印出儲存訊息
            if force:
                print(f"\n已強制儲存最新結果 (已處理 {processed_count} 次操作)")
            else:
                print(f"    已儲存中間結果 (已處理 {processed_count} 次操作)")
        
        return processed_count

    def generate_model_summary(results_data):
        """生成模型統計"""
        models_summary = {
            "專家講中文": {},
            "輕解析": {},
            "解析": {}
        }
        
        # 遍歷每個結果組
        for group, group_results in results_data.items():
            # 遍歷每個結果
            for result in group_results:
                model_name = result["model"]
                prompt_number = result["prompt_number"]
                status = "成功" if not result.get("response", "").startswith("Error") else "失敗"
                
                # 如果模型還沒有在摘要中，則初始化它
                if model_name not in models_summary[group]:
                    models_summary[group][model_name] = {
                        "成功": 0, 
                        "失敗": 0, 
                        "提示詞": [], 
                        "API時間總和": 0, 
                        "處理時間總和": 0,
                        "成功調用數": 0
                    }
                
                # 更新統計數據
                models_summary[group][model_name][status] += 1
                if prompt_number not in models_summary[group][model_name]["提示詞"]:
                    models_summary[group][model_name]["提示詞"].append(prompt_number)
                
                # 如果調用成功，則累加時間
                if status == "成功":
                    api_time = result.get("time_taken", 0)
                    models_summary[group][model_name]["API時間總和"] += api_time
                    
                    process_time = result.get("total_processing_time", 0)
                    models_summary[group][model_name]["處理時間總和"] += process_time
                    
                    models_summary[group][model_name]["成功調用數"] += 1
        
        return models_summary

    def generate_prompt_summary(results_data):
        """生成提示詞統計"""
        prompts_summary = {
            "專家講中文": {},
            "輕解析": {},
            "解析": {}
        }
        
        # 遍歷每個結果組
        for group, group_results in results_data.items():
            # 遍歷每個結果
            for result in group_results:
                model_name = result["model"]
                prompt_number = result["prompt_number"]
                status = "成功" if not result.get("response", "").startswith("Error") else "失敗"
                
                # 如果提示詞還沒有在摘要中，則初始化它
                if prompt_number not in prompts_summary[group]:
                    prompts_summary[group][prompt_number] = {"成功": 0, "失敗": 0, "模型": []}
                
                # 更新統計數據
                prompts_summary[group][prompt_number][status] += 1
                if model_name not in prompts_summary[group][prompt_number]["模型"]:
                    prompts_summary[group][prompt_number]["模型"].append(model_name)
        
        return prompts_summary

    # 初始化結果字典
    results = {
        "專家講中文": [],
        "輕解析": [],
        "解析": []
    }

    print("\n處理第一組：前20個prompt對第一組模型（專家講中文）")

    # 遍歷第一組的prompt
    for i, prompt_row in tqdm(group1_prompts.iterrows(), total=len(group1_prompts)):
        prompt_num = prompt_row.get("number", i+1)
        prompt_text = prompt_row.get("prompt", "")
        
        print(f"\n  處理 Prompt {prompt_num}...")
        
        # 遍歷第一組的模型
        for j, model in enumerate(group1_models):
            print(f"    模型 {j+1}/{len(group1_models)}: {model['name']} ({model['api_source']})...")
            
            try:
                # 記錄處理開始時間
                process_start_time = time.time()
                
                # 調用API
                result = call_llm_api(prompt_text, model, prompt_num, rules["專家講中文"])
                
                # 記錄處理結束時間並計算總處理時間
                process_end_time = time.time()
                total_processing_time = process_end_time - process_start_time
                
                result["total_processing_time"] = total_processing_time
                
                # 將結果添加到結果列表中
                results["專家講中文"].append(result)
                
                # 實時儲存結果
                save_results_realtime(results, intermediate_result_file, excel_file)
                
                # 根據API提供商設定延遲時間，避免觸發速率限制
                api_provider = result.get("api_provider", "")
                if api_provider == "openai":
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)
            except Exception as e:
                # 處理過程中發生錯誤時，記錄錯誤信息
                process_end_time = time.time()
                total_processing_time = process_end_time - process_start_time if 'process_start_time' in locals() else 0
                
                error_result = {
                    "model": model["name"],
                    "prompt_number": prompt_num,
                    "api_source": model["api_source"],
                    "api_provider": "error",
                    "response": f"Error: {str(e)}",
                    "status_code": 0,
                    "time_taken": 0,
                    "total_processing_time": total_processing_time,
                    "token_usage": {},
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "rule": rules["專家講中文"]
                }
                results["專家講中文"].append(error_result)
                print(f"    錯誤: {str(e)}")
                
                # 強制儲存結果
                save_results_realtime(results, intermediate_result_file, excel_file, force=True)
    
    # 第一組測試完成後，強制儲存一次結果
    save_results_realtime(results, intermediate_result_file, excel_file, force=True)
    print("\n第一組測試完成，已保存中間結果")

    print("\n處理第二組：中間6個prompt也對第一組模型（輕解析）")
    # 遍歷第二組的prompt
    for i, prompt_row in tqdm(group2_prompts.iterrows(), total=len(group2_prompts)):
        prompt_num = prompt_row.get("number", i+1)
        prompt_text = prompt_row.get("prompt", "")
        
        print(f"\n  處理 Prompt {prompt_num}...")
        
        # 遍歷用於測試的模型列表
        for j, model in enumerate(group2_models_for_testing):
            print(f"    模型 {j+1}/{len(group2_models_for_testing)}: {model['name']} ({model['api_source']})...")
            
            try:
                # 記錄處理開始時間
                process_start_time = time.time()
                
                # 調用API
                result = call_llm_api(prompt_text, model, prompt_num, rules["輕解析"])
                
                # 記錄處理結束時間並計算總處理時間
                process_end_time = time.time()
                total_processing_time = process_end_time - process_start_time
                
                result["total_processing_time"] = total_processing_time
                
                # 將結果添加到結果列表中
                results["輕解析"].append(result)
                
                # 實時儲存結果
                save_results_realtime(results, intermediate_result_file, excel_file)
                
                # 根據API提供商設定延遲時間
                api_provider = result.get("api_provider", "")
                if api_provider == "openai":
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)
            except Exception as e:
                # 處理過程中發生錯誤時，記錄錯誤信息
                process_end_time = time.time()
                total_processing_time = process_end_time - process_start_time if 'process_start_time' in locals() else 0
                
                error_result = {
                    "model": model["name"],
                    "prompt_number": prompt_num,
                    "api_source": model["api_source"],
                    "api_provider": "error",
                    "response": f"Error: {str(e)}",
                    "status_code": 0,
                    "time_taken": 0,
                    "total_processing_time": total_processing_time,
                    "token_usage": {},
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "rule": rules["輕解析"]
                }
                results["輕解析"].append(error_result)
                print(f"    錯誤: {str(e)}")
                
                # 強制儲存結果
                save_results_realtime(results, intermediate_result_file, excel_file, force=True)

    # 第二組測試完成後，強制儲存一次結果
    save_results_realtime(results, intermediate_result_file, excel_file, force=True)
    print("\n第二組測試完成，已保存中間結果")

    print("\n處理第三組：後20個prompt對第二組模型（解析）")
    # 遍歷第三組的prompt
    for i, prompt_row in tqdm(group3_prompts.iterrows(), total=len(group3_prompts)):
        prompt_num = prompt_row.get("number", i+1)
        prompt_text = prompt_row.get("prompt", "")
        
        print(f"\n  處理 Prompt {prompt_num}...")
        
        # 遍歷第二組的模型
        for j, model in enumerate(group2_models):
            print(f"    模型 {j+1}/{len(group2_models)}: {model['name']} ({model['api_source']})...")
            
            try:
                # 記錄處理開始時間
                process_start_time = time.time()
                
                # 調用API
                result = call_llm_api(prompt_text, model, prompt_num, rules["解析"])
                
                # 記錄處理結束時間並計算總處理時間
                process_end_time = time.time()
                total_processing_time = process_end_time - process_start_time
                
                result["total_processing_time"] = total_processing_time
                
                # 將結果添加到結果列表中
                results["解析"].append(result)
                
                # 實時儲存結果
                save_results_realtime(results, intermediate_result_file, excel_file)
                
                # 根據API提供商設定延遲時間
                api_provider = result.get("api_provider", "")
                if api_provider == "openai":
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)
            except Exception as e:
                # 處理過程中發生錯誤時，記錄錯誤信息
                process_end_time = time.time()
                total_processing_time = process_end_time - process_start_time if 'process_start_time' in locals() else 0
                
                error_result = {
                    "model": model["name"],
                    "prompt_number": prompt_num,
                    "api_source": model["api_source"],
                    "api_provider": "error",
                    "response": f"Error: {str(e)}",
                    "status_code": 0,
                    "time_taken": 0,
                    "total_processing_time": total_processing_time,
                    "token_usage": {},
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "rule": rules["解析"]
                }
                results["解析"].append(error_result)
                print(f"    錯誤: {str(e)}")
                
                # 強制儲存結果
                save_results_realtime(results, intermediate_result_file, excel_file, force=True)

    # 所有測試完成後，儲存最終的JSON結果
    final_result_file = os.path.join(output_dir, f"llm_pk_results_{timestamp}_final.json")
    with open(final_result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 將最新的Excel檔案複製一份作為最終版本
    import shutil
    shutil.copy2(excel_file, final_excel_file)

    # 印出最終結果的檔案路徑
    print(f"\n測試完成！結果已儲存至：")
    print(f"1. JSON結果: {final_result_file}")
    print(f"2. Excel結果: {final_excel_file}")

    # 印出測試執行的摘要
    print("\n=========== 測試執行摘要 ===========")
    models_summary = generate_model_summary(results)
    for group, models in models_summary.items():
        print(f"\n{group}:")
        # 根據完成率對模型進行排序並印出
        for model_name, stats in sorted(models.items(), key=lambda x: x[1]["成功"] / (x[1]["成功"] + x[1]["失敗"]) if (x[1]["成功"] + x[1]["失敗"]) > 0 else 0, reverse=True):
            total = stats["成功"] + stats["失敗"]
            completion_rate = (stats["成功"] / total * 100) if total > 0 else 0
            
            avg_api_time = (stats["API時間總和"] / stats["成功調用數"]) if stats["成功調用數"] > 0 else 0
            avg_process_time = (stats["處理時間總和"] / stats["成功調用數"]) if stats["成功調用數"] > 0 else 0
            
            print(f"  {model_name}: 成功 {stats['成功']}/{total} ({round(completion_rate, 2)}%), " +
                  f"平均API時間: {round(avg_api_time, 2)}秒, 平均總處理時間: {round(avg_process_time, 2)}秒")

    # 計算並印出總完成率
    total_success = sum(sum(stats["成功"] for stats in models.values()) for models in models_summary.values())
    total_calls = sum(sum(stats["成功"] + stats["失敗"] for stats in models.values()) for models in models_summary.values())
    total_completion_rate = (total_success / total_calls * 100) if total_calls > 0 else 0

    print(f"\n總完成率: {total_success}/{total_calls} ({round(total_completion_rate, 2)}%)")
    print("=========================================")
    
except Exception as e:
    # 捕獲並印出任何在程式執行過程中發生的錯誤
    print(f"處理過程中發生錯誤: {e}")
    import traceback
    traceback.print_exc()