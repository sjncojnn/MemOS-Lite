"""baseline — nhánh RAG thuần tuý: chunk -> embed -> vector search -> prompt -> LLM.

Độc lập hoàn toàn với src/ (MemOS-lite): không lifecycle, không cache, không provenance.
Dùng để so sánh trong scripts/eval_compare.py.
"""
