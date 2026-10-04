## 1. RAG 系統的主要種類與演進

* ### (1) Naïve RAG

  * **關鍵特徵**：基於關鍵字的檢索（如 TF-IDF、BM25），配合 LLM 做最基本的檢索生成。
  * **適用**：簡單事實型問答。
  * **限制**：無法進行語意理解、上下文弱、輸出碎片化、不易擴展。

* ### (2) Advanced RAG

  * **關鍵特徵**：引入稠密檢索（如 DPR）、神經 re-ranking、多步檢索（multi-hop）。
  * **優勢**：語意更強、適用複雜查詢、可做迭代檢索。

* ### (3) Modular RAG

  * **關鍵特徵**：檢索/生成模組化、支持混合檢索（Sparse+Dense）、可插拔工具（API、DB）。
  * **優勢**：高彈性、可擴展、可根據任務客製工作流。

* ### (4) Graph RAG

  * **關鍵特徵**：引入圖結構（knowledge graph），支援 entity、關聯、階層推理。
  * **適用**：結構化知識推理（如醫療、法務）。
  * **限制**：資料依賴性高，系統較複雜。

* ### (5) Agentic RAG

  * **關鍵特徵**：引入自主 agent，可做動態決策、流程優化、多步自我修正（reflection、planning、tool use、multi-agent）。
  * **優勢**：動態適應、多代理合作、可進行複雜推理、流程自動調整。
  * **挑戰**：協調與資源管理難度提高。

| Paradigm<br>範式             | Key Features<br>關鍵特徵                                                                                                                                  | Strengths<br>優勢                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Naïve RAG**<br>基礎型RAG    | - Keyword-based retrieval (e.g., TF-IDF, BM25)<br>基於關鍵字的檢索                                                                                            | - Simple and easy to implement<br>簡單易上手<br>- Suitable for fact-based queries<br>適用事實型查詢                                               |
| **Advanced RAG**<br>進階型RAG | - Dense retrieval models (e.g., DPR)<br>稠密檢索模型<br>- Neural ranking and re-ranking<br>神經排序與重排<br>- Multi-hop retrieval<br>多跳檢索                         | - High precision retrieval<br>高精度檢索<br>- Improved contextual relevance<br>更佳語境相關性                                                     |
| **Modular RAG**<br>模組化RAG  | - Hybrid retrieval (sparse and dense)<br>稀疏/稠密混合檢索<br>- Tool and API integration<br>整合工具與API<br>- Composable, domain-specific pipelines<br>可組合化、領域化流程 | - High flexibility and customization<br>高彈性與客製化<br>- Suitable for diverse applications<br>適用多元應用<br>- Scalable<br>易擴展                 |
| **Graph RAG**<br>圖結構RAG    | - Integration of graph-based structures<br>整合圖結構<br>- Multi-hop reasoning<br>多跳推理<br>- Contextual enrichment via nodes<br>節點增強語境                      | - Relational reasoning capabilities<br>關聯推理能力<br>- Mitigates hallucinations<br>減少幻覺<br>- Ideal for structured data tasks<br>結構化資料任務最佳 |
| **Agentic RAG**<br>智能代理RAG | - Autonomous agents<br>自主代理<br>- Dynamic decision-making<br>動態決策<br>- Iterative refinement and workflow optimization<br>多輪優化與流程調度                     | - Adaptable to real-time changes<br>可因應即時變化<br>- Scalable for multi-domain tasks<br>多領域易擴展<br>- High accuracy<br>高準確度                 |


---

## 2. Agent（智能代理）的設計組件與核心 pattern

* **LLM（主體引擎）**：負責理解查詢、生成、維持對話一致性。
* **記憶模組（短期、長期）**：可追蹤多輪上下文、累積知識。
* **規劃（Planning、Reflection）**：可做任務拆解、自省、自我批評和修正。
* **工具調用（Tool Use）**：API、外部 DB、Web Search 等整合。
* **多代理協作（Multi-Agent Collaboration）**：多個 agent 分工合作，分層處理。

---

## 3. Agentic Workflow Patterns（Agentic 工作流模式）

* **Prompt Chaining**：任務拆解成連續步驟，逐步推理增強正確性。
* **Routing**：根據查詢類型導向不同子工作流（如簡單查詢導向小模型，複雜查詢導向專家模型）。
* **Parallelization**：多路同時執行（如分段、投票）。
* **Orchestrator-Worker**：中心調度 agent 動態分派、整合結果（可動態調整，適合複雜查詢）。
* **Evaluator-Optimizer**：生成-評估-再生成-優化，透過多輪修正提升品質。
* **Corrective RAG**：自動判斷文件是否相關，不相關就自動修正查詢、再檢索。
* **Adaptive Agentic RAG**：根據查詢複雜度自動選擇最佳處理路徑（直接回覆、單步檢索、多步推理等）。
* **Graph-Based Agentic RAG**：結合圖結構與 agentic workflow，可處理複雜關係、多步查詢。
* **Agentic Document Workflows (ADW)**：以文件為中心的自動化處理，涵蓋解析、檢索、推理、輸出。

---

## 4. Agentic RAG 架構分類（Taxonomy）

* ### (1) Single-Agent Agentic RAG

  * 一個主代理負責檢索、路由、整合。
  * 適合簡單工作流、工具數少的場合。

* ### (2) Multi-Agent Agentic RAG

  * 多代理各司其職（如 SQL agent、Web Search agent、Semantic agent 等），由主調度協調。
  * 適合複雜查詢、多資料源、跨領域場景。

* ### (3) Hierarchical Agentic RAG

  * 多層代理（如 Top-tier、Mid-level、Lower-level），上下分層管理與協作。
  * 適合需要分層決策、動態分工的高複雜度場景（如金融、醫療、合約分析）。

* ### (4) Corrective/Adaptive/Graph-based Agentic RAG

  * 強調流程動態修正、自我優化、圖結構推理等特殊應用。

* ### (5) Agentic Document Workflows (ADW)

  * 以「文件」為核心的多代理自動化，如合約審查、發票自動處理等。

| Feature<br>功能面向                             | Traditional RAG<br>傳統 RAG                            | Agentic RAG<br>智能代理 RAG                                      | Agentic Document Workflows (ADW)<br>文件導向智能代理工作流                             |
| ------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| **Focus<br>焦點**                             | Isolated retrieval and generation tasks<br>單一檢索與生成任務 | Multi-agent collaboration and reasoning<br>多代理協作與推理          | Document-centric end-to-end workflows<br>文件為核心的端到端工作流                       |
| **Context Maintenance<br>上下文維護**            | Limited<br>有限                                        | Enabled through memory modules<br>靠記憶模組強化                    | Maintains state across multi-step workflows<br>多步流程狀態持續追蹤                   |
| **Dynamic Adaptability<br>動態適應性**           | Minimal<br>極低                                        | High<br>高度動態                                                 | Tailored to document workflows<br>針對文件流程優化                                  |
| **Workflow Orchestration<br>流程協作**          | Absent<br>無                                          | Orchestrates multi-agent tasks<br>協調多代理流程                    | Integrates multi-step document processing<br>整合多步文件處理                       |
| **Use of External Tools/APIs<br>外部工具/接口應用** | Basic integration (e.g., retrieval tools)<br>僅基礎工具   | Extends via tools like APIs and knowledge bases<br>整合API/知識庫 | Deeply integrates business rules and domain-specific tools<br>深度整合商業規則與專屬工具 |
| **Scalability<br>可擴展性**                     | Limited to small datasets or queries<br>只適合小規模       | Scalable for multi-agent systems<br>多代理可擴展                   | Scales for multi-domain enterprise workflows<br>多領域企業級可擴展                   |
| **Complex Reasoning<br>複雜推理**               | Basic (e.g., simple Q\&A)<br>僅基礎問答                   | Multi-step reasoning with agents<br>多代理多步推理                  | Structured reasoning across documents<br>跨文件結構化推理                           |
| **Primary Applications<br>主要應用**            | QA systems, knowledge retrieval<br>問答與知識檢索           | Multi-domain knowledge and reasoning<br>多領域知識推理              | Contract review, invoice processing, claims analysis<br>合約審查、發票處理、理賠分析      |
| **Strengths<br>優勢**                         | Simplicity, quick setup<br>簡單好部署                     | High accuracy, collaborative reasoning<br>高準確協作推理            | End-to-end automation, domain-specific intelligence<br>全自動化、領域專業            |
| **Challenges<br>挑戰**                        | Poor contextual understanding<br>上下文理解弱              | Coordination complexity<br>協作與管理複雜度高                         | Resource overhead, domain standardization<br>資源開銷與領域標準化                     |

---
# Benchmarks and Task Types
| Category<br>類別                  | Task Type<br>任務型態                       | Datasets and References<br>資料集與參考                                              |
| ------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------ |
| **QA<br>問答**                    | Single-hop QA<br>單跳問答                   | Natural Questions (NQ), TriviaQA, SQuAD, Web Questions (WebQ), PopQA, MS MARCO |
|                                 | Multi-hop QA<br>多跳問答                    | HotpotQA, 2WikiMultiHopQA, MuSiQue                                             |
|                                 | Long-form QA<br>長篇問答                    | ELI5, NarrativeQA (NQA), ASQA, QMSum                                           |
|                                 | Domain-specific QA<br>領域問答              | Qasper, COVID-QA, CMB/MMCU Medical                                             |
|                                 | Multi-choice QA<br>多選問答                 | QuALITY, ARC, CommonsenseQA                                                    |
| **Graph-based QA<br>圖問答**       | Graph QA<br>圖推理問答                       | GraphQA                                                                        |
|                                 | Event Argument Extraction<br>事件論據擷取     | WikiEvent, RAMS                                                                |
| **Dialog<br>對話**                | Open-domain Dialog<br>開放領域對話            | Wizard of Wikipedia (WoW)                                                      |
|                                 | Personalized Dialog<br>個人化對話            | KBP, DuleMon                                                                   |
|                                 | Task-oriented Dialog<br>任務型對話           | CamRest                                                                        |
| **Recommendation<br>推薦**        | Personalized Content<br>個人化推薦           | Amazon Datasets (Toys, Sports, Beauty)                                         |
| **Reasoning<br>推理**             | Commonsense Reasoning<br>常識推理           | HellaSwag, CommonsenseQA                                                       |
|                                 | CoT Reasoning<br>鏈式推理                   | CoT Reasoning                                                                  |
|                                 | Complex Reasoning<br>複雜推理               | CSQA                                                                           |
| **Others<br>其他**                | Language Understanding<br>語言理解          | MMLU, WikiText-103                                                             |
|                                 | Fact Checking/Verification<br>事實查核      | FEVER, PubHealth                                                               |
|                                 | Strategy QA<br>策略問答                     | StrategyQA                                                                     |
| **Summarization<br>摘要**         | Text Summarization<br>文本摘要              | WikiASP, XSum                                                                  |
|                                 | Long-form Summarization<br>長摘要          | NarrativeQA (NQA), QMSum                                                       |
| **Text Generation<br>文本生成**     | Biography<br>傳記                         | Biography Dataset                                                              |
| **Text Classification<br>分類**   | Sentiment Analysis<br>情感分析              | SST-2                                                                          |
|                                 | General Classification<br>一般分類          | Violens, TREC                                                                  |
| **Code Search<br>程式搜尋**         | Programming Search<br>程式檢索              | CodeSearchNet                                                                  |
| **Robustness<br>魯棒性**           | Retrieval Robustness<br>檢索魯棒性           | NoMIRACL                                                                       |
|                                 | Language Modeling Robustness<br>語言模型魯棒性 | WikiText-103                                                                   |
| **Math<br>數學**                  | Math Reasoning<br>數學推理                  | GSM8K                                                                          |
| **Machine Translation<br>機器翻譯** | Translation Tasks<br>翻譯任務               | JRC-Acquis                                                                     |

