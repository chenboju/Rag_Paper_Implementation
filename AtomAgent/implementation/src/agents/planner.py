import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.core.state import AgentState
from src.agents.prompts import PLANNER_SYSTEM_PROMPT

# 1. 定義結構化輸出 (這是關鍵！)
# 我們強迫 LLM 必須回傳這個 Class 的格式
class ResearchPlan(BaseModel):
    """The list of steps to solve the research problem."""
    steps: List[str] = Field(description="List of sequential steps to follow.")

class PlannerAgent:
    def __init__(self):
        # 初始化 LLM
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4.1"),
            temperature=0
        )
        # 綁定結構化輸出，強轉型為 ResearchPlan 物件
        self.planner_llm = llm.with_structured_output(ResearchPlan)

    def run(self, state: AgentState):
        """
        Planner 的節點函數。
        輸入: 使用者的訊息
        輸出: 更新 State 中的 'plan' 欄位
        """
        print("🧠 Planner is thinking...")
        
        # 1. 準備 Prompt
        # 我們只看使用者的第一條訊息 (需求)，或者最新的對話
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])
        
        # 2. 執行推論
        chain = prompt | self.planner_llm
        plan_result: ResearchPlan = chain.invoke({"messages": messages})
        
        # 3. 顯示計畫 (Debug 用)
        print(f"📋 Plan Generated: {len(plan_result.steps)} steps")
        for i, step in enumerate(plan_result.steps):
            print(f"   {i+1}. {step}")

        # 4. 回傳更新狀態
        # 注意: 這裡我們不回傳 messages，只更新 'plan'
        return {"plan": plan_result.steps}