import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.core.state import AgentState
from src.agents.prompts import RESULT_CRITIC_SYSTEM_PROMPT

class ResultReview(BaseModel):
    decision: Literal["approve", "reject"] = Field(description="Decision on the simulation result.")
    feedback: str = Field(description="Feedback for the Engineer if rejected.")

class ResultCriticAgent:
    def __init__(self):
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4.1"),
            temperature=0
        )
        self.llm = llm.with_structured_output(ResultReview)

    def run(self, state: AgentState):
        print("🧐 Result Critic is reviewing the output...")
        
        # 1. 讀取 Engineer 最新的執行結果 (對話歷史)
        messages = state["messages"]
        
        # 2. 構建 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESULT_CRITIC_SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])
        
        # 3. 執行
        chain = prompt | self.llm
        review: ResultReview = chain.invoke({"messages": messages})
        
        print(f"   (Result Critic) Verdict: {review.decision.upper()}")
        
        # 4. 回傳狀態 (只更新 result 相關欄位)
        # 如果駁回，將 feedback 加入對話歷史給 Engineer 看
        return {
            "critic_decision": review.decision,
            "result_feedback": review.feedback,
            "messages": [review.feedback] if review.decision == "reject" else []
        }