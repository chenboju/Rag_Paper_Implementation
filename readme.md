# RAG Research & Paper Implementations

A curated research workspace for **Retrieval-Augmented Generation (RAG)**, **Agentic AI**, **materials-science agents**, and paper reproduction experiments.

This repository consolidates the reusable parts of several earlier experimental repositories into one structured research codebase.

## Research Areas

| Area | Purpose | Location |
| --- | --- | --- |
| AtomAgent | LangGraph/LangChain agent workflow with RAG and LAMMPS-related tools | `AtomAgent/` |
| GNoME | Materials-discovery paper study and presentation assets | `GNoME/` |
| SimRAG | SimRAG paper notes and architecture assets | `SimRAG/` |
| RAG Evaluation | Legacy RAGAS evaluation notebooks and QA experiments | `Evaluation/` |
| Research Notes | Selected RAG, Agentic RAG, Generative QA, OCR and MinerU experiments | `ResearchNotes/` |

## Repository Structure

```text
.
├── AtomAgent/
│   ├── 2407.10022v1.pdf
│   ├── AtomAgents.pptx
│   └── implementation/
│       ├── src/
│       │   ├── agents/
│       │   ├── core/
│       │   ├── engine/
│       │   └── tools/
│       ├── tests/
│       └── requirements.txt
│
├── GNoME/
├── SimRAG/
│
├── Evaluation/
│   └── RAGAS-legacy/
│
└── ResearchNotes/
    └── legacy-rag-notes/
        ├── Generative QA/
        ├── OCR/
        └── Paper/
```

## AtomAgent Implementation

The `AtomAgent/implementation` directory contains the reusable implementation migrated from the earlier `AtomLangchainGraphAgent` project.

Core components include:

- LangGraph-based agent orchestration
- planner / engineer / critic agents
- RAG retrieval tools
- physics and plotting tools
- LAMMPS execution and parsing utilities
- structure generation
- tests and dependency definitions

Generated simulation logs, duplicated document corpora, editor settings, and temporary output files were intentionally excluded.

## RAG Evaluation

`Evaluation/RAGAS-legacy` preserves earlier experiments involving:

- RAGAS evaluation
- local/global RAG experiments
- QA test sets
- consumer-protection-law QA data

These experiments are retained as historical baselines rather than treated as the current evaluation pipeline.

## Research Notes

`ResearchNotes/legacy-rag-notes` contains selected reusable material from earlier RAG study work, including:

- Generative QA scripts
- SimRAG experiments
- Agentic RAG notes
- MinerU / OCR processing scripts

Large presentation collections, duplicated notebooks, and transient artifacts were intentionally not copied.

## Consolidation History

| Historical repository | Current destination |
| --- | --- |
| `AtomLangchainGraphAgent` | `AtomAgent/implementation/` |
| `RAG-Evaluation` | `Evaluation/RAGAS-legacy/` |
| `Large-language-model-application-RAG` | `ResearchNotes/legacy-rag-notes/` |

The historical repositories are retained as source snapshots. New related work should be organized in this repository instead of creating another standalone legacy RAG repository.

## Status

This repository is the primary location for ongoing paper implementation and RAG research organization.
