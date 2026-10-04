import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.state import AgentState
from src.agents.prompts import ENGINEER_SYSTEM_PROMPT
from src.tools.physics_tools import calculate_lattice_constant

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()


# only enginner邏輯
# class EngineerAgent:
#     def __init__(self):
#         # 1. 初始化模型
#         # 雖然 Embedding 用 4090，但邏輯推理我們用 GPT-4o，因為它的 Function Calling 能力最強
#         self.llm = ChatOpenAI(
#             model=os.getenv("OPENAI_MODEL_NAME", "gpt-4.1"),
#             temperature=0  # 設為 0 讓行為更穩定，適合寫程式/科學計算
#         )
        
#         # 2. 準備工具箱
#         self.tools = [calculate_lattice_constant]
        
#         # 3. 綁定工具 (Bind Tools)
#         # 這一步是關鍵：它會把 Python 函數的參數定義轉換成 OpenAI 懂的 JSON Schema
#         self.llm_with_tools = self.llm.bind_tools(self.tools)

    # only enginner邏輯
    # def run(self, state: AgentState):
    #     """
    #     這是 LangGraph 的節點函數。
    #     它接收當前的 State，進行思考，然後回傳新的訊息。
    #     """
    #     # 提取對話歷史
    #     messages = state["messages"]
        
    #     # 建立 Prompt
    #     # MessagesPlaceholder("messages") 會把過去的對話紀錄塞進去
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", ENGINEER_SYSTEM_PROMPT),
    #         MessagesPlaceholder(variable_name="messages"),
    #     ])
        
    #     # 建立 Chain: Prompt -> LLM (帶有工具)
    #     chain = prompt | self.llm_with_tools
        
    #     # 執行推論
    #     print("🤖 Engineer is thinking...")
    #     response = chain.invoke({"messages": messages})
        
    #     # 回傳結果
    #     # 因為我們在 State 定義了 operator.add，這個 response 會被自動 append 到 messages 列表末端
    #     return {"messages": [response]}

class EngineerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            temperature=0
        )
        self.tools = [calculate_lattice_constant]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, state: AgentState):
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        
        current_task = "Respond to user."
        if plan and index < len(plan):
            current_task = plan[index]
            print(f"👷 Engineer working on Step {index+1}: {current_task}")
        
        # --- [修正點開始] ---
        # 1. 定義 Prompt Template 時，使用 {變數名稱}，不要直接用 f-string 塞值
        prompt = ChatPromptTemplate.from_messages([
            ("system", ENGINEER_SYSTEM_PROMPT),
            ("system", "Current Execution Step: {current_task}"), 
            ("system", "Context from previous steps: {context_data}, Error count: {error_count}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | self.llm_with_tools
        
        # 2. 準備要傳入的變數
        # 將字典轉為字串，這樣 LangChain 就不會再去解析裡面的括號了
        context_str = str(state.get('simulation_results', {}))
        error_cnt = state.get('error_count', 0)

        # 3. 執行 Invoke，傳入所有變數
        response = chain.invoke({
            "messages": state["messages"],
            "current_task": current_task,
            "context_data": context_str,
            "error_count": error_cnt
        })
        # --- [修正點結束] ---
        
        return {"messages": [response]}