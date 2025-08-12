# A Two-Stage Automated Evaluation Framework for AI-Generated Equity Research

**Project Type:** Core Internship Project at **Wistock Company (Fintech Startup)**.

**Status:** Completed and deployed in the company's internal model evaluation pipeline.

---

## 1. Objective & Motivation

With the exponential growth of Large Language Models (LLMs), selecting the optimal model for generating high-quality financial analysis has become a critical challenge for Fintech companies. A purely subjective, manual evaluation is neither scalable nor objective.

This project was initiated to solve this problem by architecting and building a **robust, automated, and quantitative framework** to systematically benchmark LLM performance for equity research tasks. The primary goal was to provide the company with a data-driven methodology to select the most accurate, coherent, and cost-effective model for its core products.

---

## 2. System Architecture & Implementation

I designed and implemented a sophisticated **two-stage evaluation pipeline** entirely in Python, leveraging asynchronous programming to handle large-scale API calls efficiently.

### Stage 1: Mass-Scale Response Generation (`llm_pk_api.py`)

The first stage is a powerful and flexible API abstraction layer designed to gather responses from a wide array of LLM providers.

* **Multi-Provider API Integration:** The system seamlessly interfaces with multiple major LLM providers, including **OpenAI, OpenRouter, and Groq**, allowing for parallel testing of dozens of models (e.g., GPT series, Claude series, Llama, Mistral).
* **Automated Task Management:** The entire process is driven by an Excel input file, which defines the models to be tested, the prompts to be used, and the API keys, enabling non-technical team members to easily configure and run new evaluation tasks.
* **Real-time & Resilient Data Handling:** Utilized `tqdm` for progress tracking and implemented a robust retry mechanism to handle API errors, with all results, including performance metrics (latency, token usage) and timestamps, saved in real-time to prevent data loss.

### Stage 2: AI-Powered Quantitative Evaluation (`process_evaluations.py`)

The second stage is a novel **AI-powered evaluation pipeline** that uses top-tier LLMs as "judges" to score the quality of the responses gathered in Stage 1. This "AI evaluating AI" approach provides a scalable and objective scoring mechanism.

* **Asynchronous Evaluation Engine:** Leveraged Python's `asyncio` library and a `Semaphore` to manage a high-concurrency request queue (up to 100 concurrent requests), drastically reducing the time required for evaluation.
* **Multi-Dimensional Scoring Rubric:** I designed and implemented a detailed, four-dimensional scoring rubric that instructs the "judge" LLMs (e.g., GPT-4o, Claude 3.5 Sonnet) to act as expert financial analysts and score each report based on the following quantitative criteria:
    1.  **Content Fidelity (10%):** Absolute adherence to the data and facts provided in the prompt.
    2.  **Financial Proficiency (10%):** Correct understanding and application of financial data and terminology.
    3.  **Logical Structure (40%):** The logical flow, coherence, and persuasiveness of the argument.
    4.  **Linguistic Expression (40%):** The clarity, fluency, and professionalism of the generated Chinese text.

***Note: Due to the proprietary nature of the work, the full source code and data files cannot be made public. This repository contains key code snippets, methodology explanations, and serves as a detailed description of the project's architecture and impact. The main scripts `llm_pk_api.py` and `process_evaluations.py` are provided as a demonstration of the system's logic and my coding style.***

---

## 3. Impact & Contribution

* **Data-Driven Strategy:** This framework provided the company with an objective, quantitative basis to select the optimal LLM for its products, directly influencing key technical and business decisions.
* **Massive Efficiency Gains:** The fully automated pipeline replaced a slow, subjective manual evaluation process, enabling the team to benchmark dozens of models against hundreds of prompts in a fraction of the time.
* **Advanced Engineering Skills:** This project demonstrates mastery of advanced Python programming (including `asyncio`), complex system architecture design, multi-provider API integration, and the innovative application of AI for automated quality assessment.

---

## 4. Key Technologies & Tools

* **Languages & Libraries:** Python (Pandas, Asyncio, Requests, OpenPyXL)
* **Core Fields:** AI Engineering, System Architecture, MLOps, Fintech
* **APIs & Platforms:** OpenAI, OpenRouter, Groq
