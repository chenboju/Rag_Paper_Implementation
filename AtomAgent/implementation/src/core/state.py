import operator
from typing import Annotated, List, Dict, Any, TypedDict, Union
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    這是所有 Agent 共用的記憶體結構。
    """
    # 1. messages: 對話歷史
    # Annotated[..., operator.add] 的意思是：
    # 當有新的訊息進來時，不要「覆蓋」舊的，而是「附加 (Append)」到列表後面。
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 2. plan: 任務清單 (未來由 Planner 生成，目前先預留)
    plan: List[str]
    current_step_index: int
    simulation_results: Dict[str, Any]
    error_count: int

    # 3. current_step: 讓 Agent 知道現在執行到第幾步
    current_step_index: int
    
    # 4. simulation_results: 存放物理模擬的關鍵數據 (例如: {"lattice_constant": 3.30})
    # 這讓我們不需要把幾 MB 的 Log 檔塞給 LLM，只給它看結果。
    simulation_results: Dict[str, Any]
    # --- Plan Critic 專用 (互不影響) ---
    plan_valid: bool        # 計畫是否通過
    plan_feedback: str      # 給 Planner 的修改建議
    
    # --- Result Critic 專用 (互不影響) ---
    critic_decision: str    # 'approve' 或 'reject'
    result_feedback: str    # 給 Engineer 的修改建議 (若駁回)