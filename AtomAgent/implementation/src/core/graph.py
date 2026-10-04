
# # only enginner 部分
# # from langgraph.graph import StateGraph, END
# # from langgraph.prebuilt import ToolNode

# # from src.core.state import AgentState
# # from src.agents.engineer import EngineerAgent
# # from src.tools.physics_tools import calculate_lattice_constant

# # # 1. 實例化 Agent
# # engineer_agent = EngineerAgent()

# # # 2. 準備工具節點 (ToolNode)
# # # LangGraph 提供了一個現成的 ToolNode，它會自動讀取 LLM 回傳的 'tool_calls'，
# # # 執行對應的 Python 函數，並將結果包裝成 'ToolMessage' 回傳。
# # tools = [calculate_lattice_constant]
# # tool_node = ToolNode(tools)

# # # 3. 定義路由邏輯 (Router)
# # def should_continue(state: AgentState):
# #     """
# #     這是一個「紅綠燈」函數。
# #     它檢查最後一條訊息，看看 LLM 是否想要呼叫工具。
# #     """
# #     last_message = state["messages"][-1]
    
# #     # 如果訊息裡包含 tool_calls，代表 LLM 想用工具 -> 轉接給 'tools' 節點
# #     if last_message.tool_calls:
# #         return "tools"
    
# #     # 否則，代表 LLM 已經講完話了 -> 結束 (END)
# #     return END

# # # 4. 構建圖表 (Graph Construction)
# # workflow = StateGraph(AgentState)

# # # (A) 加入節點
# # workflow.add_node("engineer", engineer_agent.run)
# # workflow.add_node("tools", tool_node)

# # # (B) 設定起點
# # # 系統一啟動，先進入 Engineer 節點思考
# # workflow.set_entry_point("engineer")

# # # (C) 加入條件邊 (Conditional Edge)
# # # 從 engineer 出發，執行 should_continue 函數
# # # 如果回傳 "tools" -> 走到 "tools" 節點
# # # 如果回傳 END -> 流程結束
# # workflow.add_conditional_edges(
# #     "engineer",
# #     should_continue,
# #     {
# #         "tools": "tools",
# #         END: END
# #     }
# # )

# # # (D) 加入普通邊 (Normal Edge)
# # # 工具執行完後，必須把結果拿回去給 Engineer 看，讓他解釋結果
# # workflow.add_edge("tools", "engineer")

# # # 5. 編譯 (Compile)
# # # 這會把上面的定義轉換成一個可執行的應用程式
# # app = workflow.compile()


# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# from src.core.state import AgentState
# from src.agents.planner import PlannerAgent
# from src.agents.engineer import EngineerAgent
# from src.tools.physics_tools import calculate_lattice_constant

# # 1. 初始化 Agents
# planner_agent = PlannerAgent()
# engineer_agent = EngineerAgent()

# # 2. 工具節點
# tools = [calculate_lattice_constant]
# tool_node = ToolNode(tools)

# # 3. 條件判斷：Engineer 執行完後該去哪？
# def route_engineer(state: AgentState):
#     """
#     決定 Engineer 之後的去向：
#     1. 想用工具 -> 去 Tools
#     2. 講完話了 -> 檢查計畫進度 (check_plan)
#     """
#     last_message = state["messages"][-1]
#     if last_message.tool_calls:
#         return "tools"
#     return "check_plan"

# # 4. 條件判斷：計畫執行完了嗎？
# def check_plan_status(state: AgentState):
#     """
#     檢查是否還有下一個步驟。
#     """
#     plan = state.get("plan", [])
#     index = state.get("current_step_index", 0)
    
#     # 如果 Engineer 剛完成了一個步驟 (回傳了文字結果)
#     # 我們讓索引 +1
#     next_index = index + 1
    
#     if next_index < len(plan):
#         # 還有下一步 -> 更新索引，回到 Engineer
#         print(f"🔄 Proceeding to Step {next_index + 1}/{len(plan)}")
#         return "next_step"
#     else:
#         # 沒有下一步 -> 結束
#         print("✅ All steps completed.")
#         return END

# # 5. 更新狀態用的簡單節點 (用於 index + 1)
# def update_step_node(state: AgentState):
#     return {"current_step_index": state["current_step_index"] + 1}

# # --- 構建 Graph ---
# workflow = StateGraph(AgentState)

# # 加入節點
# workflow.add_node("planner", planner_agent.run)
# workflow.add_node("engineer", engineer_agent.run)
# workflow.add_node("tools", tool_node)
# workflow.add_node("update_step", update_step_node) # 負責 index++

# # 設定流程
# # 1. 起點 -> Planner
# workflow.set_entry_point("planner")

# # 2. Planner -> Engineer (開始執行第一步)
# workflow.add_edge("planner", "engineer")

# # 3. Engineer 的路由
# workflow.add_conditional_edges(
#     "engineer",
#     route_engineer,
#     {
#         "tools": "tools",
#         "check_plan": "update_step" # 如果 Engineer 說話了，代表這一步做完了，去更新索引
#     }
# )

# # 4. Tools -> Engineer (工具跑完，回報給 Engineer)
# workflow.add_edge("tools", "engineer")

# # 5. Update Step -> 判斷是否結束
# workflow.add_conditional_edges(
#     "update_step",
#     check_plan_status,
#     {
#         "next_step": "engineer", # 還有步驟，回去 Engineer
#         END: END                 # 做完了
#     }
# )

# app = workflow.compile()


#加入Critic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 導入 Agents ---
from src.core.state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.engineer import EngineerAgent
from src.agents.plan_critic import PlanCriticAgent
from src.agents.result_critic import ResultCriticAgent

# --- 導入工具 (Tools) ---
from src.tools.rag_tools import retrieve_knowledge
# [新增] 導入新的物理視覺化工具
from src.tools.plot_tools import plot_and_save_data, generate_dd_map, generate_neb_plot
from src.tools.physics_tools import calculate_lattice_constant

# 1. 初始化 Agents
planner = PlannerAgent()
engineer = EngineerAgent()
plan_critic = PlanCriticAgent()
result_critic = ResultCriticAgent()

# 2. 定義工具列表
# [關鍵修改] 將新工具加入此列表
tools = [
    calculate_lattice_constant, 
    retrieve_knowledge, 
    plot_and_save_data,
    generate_dd_map,    # 新增: 位錯視覺化
    generate_neb_plot   # 新增: NEB 能量路徑圖
]

# 重新綁定 ToolNode
tool_node = ToolNode(tools)

# --- 路由邏輯 ---

def route_plan_critic(state: AgentState):
    """Plan Critic 的路由：決定計畫通過與否"""
    if state.get("plan_valid", False):
        return "engineer" # 通過 -> 交給 Engineer
    else:
        return "planner"  # 駁回 -> 退回 Planner 重寫

def route_engineer(state: AgentState):
    """Engineer 的路由：用工具還是送審"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "result_critic" # 做完了 -> 送去 Result Critic

def route_result_critic(state: AgentState):
    """Result Critic 的路由：決定結果通過與否"""
    if state.get("critic_decision") == "approve":
        return "update_step" # 通過 -> 更新步驟索引
    return "engineer"        # 駁回 -> Engineer 重做

def check_plan_status(state: AgentState):
    """檢查是否還有下一步驟"""
    plan = state.get("plan", [])
    index = state.get("current_step_index", 0)
    if index < len(plan):
        return "next_step"
    return END

def update_step_node(state: AgentState):
    return {"current_step_index": state["current_step_index"] + 1}

# --- 構建 Graph ---
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("planner", planner.run)
workflow.add_node("plan_critic", plan_critic.run)
workflow.add_node("engineer", engineer.run)
workflow.add_node("result_critic", result_critic.run)
workflow.add_node("tools", tool_node)
workflow.add_node("update_step", update_step_node)

# Edges & Routing
workflow.set_entry_point("planner")

# 1. 規劃迴圈
workflow.add_edge("planner", "plan_critic")
workflow.add_conditional_edges(
    "plan_critic",
    route_plan_critic,
    {"engineer": "engineer", "planner": "planner"}
)

# 2. 執行迴圈
workflow.add_conditional_edges(
    "engineer",
    route_engineer,
    {"tools": "tools", "result_critic": "result_critic"}
)
workflow.add_edge("tools", "engineer")

# 3. 結果審查迴圈
workflow.add_conditional_edges(
    "result_critic",
    route_result_critic,
    {"update_step": "update_step", "engineer": "engineer"}
)

# 4. 總流程控制
workflow.add_conditional_edges(
    "update_step",
    check_plan_status,
    {"next_step": "engineer", END: END}
)

app = workflow.compile()