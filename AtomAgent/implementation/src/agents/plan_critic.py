import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.core.state import AgentState
from src.agents.prompts import PLAN_CRITIC_SYSTEM_PROMPT

# --- 1. 定義輸出結構 (這就是報錯說缺少的 PlanReview) ---
class PlanReview(BaseModel):
    decision: Literal["approve", "reject"] = Field(description="Decision on the plan.")
    feedback: str = Field(description="Feedback if rejected, empty if approved.")

# --- 2. 定義 Agent ---
class PlanCriticAgent:
    def __init__(self):
        # 使用獨立的模型實例
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            temperature=0
        )
        self.llm = llm.with_structured_output(PlanReview)

    def run(self, state: AgentState):
        print("🕵️ Plan Critic is reviewing the strategy...")
        
        # 1. 準備資料
        current_plan = state.get("plan", [])
        plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(current_plan)])
        
        # 防呆：確保 messages 不為空
        user_request = "No user request found."
        if state["messages"]:
            user_request = state["messages"][0].content
        
        # 2. 準備 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLAN_CRITIC_SYSTEM_PROMPT),
            ("user", f"User Request: {user_request}\n\nProposed Plan:\n{plan_str}")
        ])
        
        # 3. 執行 Chain
        # 修正點：必須使用 invoke({}) 來觸發執行，即使沒有變數也要傳空字典
        chain = prompt | self.llm
        review: PlanReview = chain.invoke({})
        
        print(f"   (Plan Critic) Verdict: {review.decision.upper()}")
        if review.decision == "reject":
            print(f"   (Plan Critic) Feedback: {review.feedback}")

        # 4. 回傳狀態
        return {
            "plan_valid": (review.decision == "approve"),
            "plan_feedback": review.feedback
        }