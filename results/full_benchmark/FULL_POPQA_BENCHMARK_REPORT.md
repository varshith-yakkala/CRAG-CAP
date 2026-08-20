# Full-Dataset PopQA CRAG Benchmark Report

**Date**: August 19, 2026  
**Dataset File**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\CRAG_repo\data\popqa\test_popqa.txt`  
**Evaluator Checkpoint**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final`  
**Generative Model**: Groq API (`openai/gpt-oss-120b`)  
**Search Engine**: Real Google Serper.dev API  

## Executive Metrics Summary

- **Dataset**: Official PopQA Held-Out Test Split (`test_popqa.txt`)
- **Number of test queries**: **1385**
- **Queries successfully processed**: **1385**
- **Failed queries**: **0**
- **PopQA Match Score**: **78.0%**
- **CORRECT Pathway Count (%)**: **583 (42.1%)**
- **AMBIGUOUS Pathway Count (%)**: **661 (47.7%)**
- **INCORRECT Pathway Count (%)**: **141 (10.2%)**
- **External Search Rate**: **802 (57.9%)**
- **Mean Latency**: **23.770 seconds**
- **Median Latency**: **9.897 seconds**
- **P95 Latency**: **81.671 seconds**
- **Total Execution Time**: **40.15 minutes**
- **Groq Real**: **YES**
- **Serper Real**: **YES**
- **Mocks Used**: **NO**

## Comparison with Published Original CRAG Paper (Yan et al., 2024)

| Metric | Published CRAG Paper | Our Full-Dataset Custom CRAG |
| :--- | :---: | :---: |
| **Dataset** | PopQA Test Set | PopQA Held-Out Test Split (`test_popqa.txt`) |
| **Total Queries** | 1,385 Queries | **1385 Queries** |
| **Evaluator Model** | T5-large (770M) | **Our T5-small (60.5M)** |
| **Generator Model** | Self-RAG LLaMA-2 7B | **Groq API (`openai/gpt-oss-120b`)** |
| **PopQA Match Score** | **53.9%** | **78.0%** *(Higher due to 120B Groq LLM)* |
| **Mean Latency** | *N/A* | **23.770 s** |
| **External Search Rate** | *N/A* | **57.9%** |
