# LoRA fine-tuning, RAG from scratch and running a model locally
 
Three Google Colab notebooks, one per task. Each is written to run top to bottom on a free T4 GPU and installs its own dependencies, so there is nothing to set up locally.
 
| Task | Notebook | What it does |
|---|---|---|
| 1. Fine-tuning | [`task1_lora_finetune.ipynb`](task1_lora_finetune (1).ipynb) | LoRA-fine-tunes Qwen2.5-0.5B-Instruct to label a tweet with one of six emotions, and compares base and fine-tuned accuracy on about 300 held-out tweets. |
| 2. Retrieval (RAG) | [`task2_rag_from_scratch.ipynb`](task2_rag_from_scratch.ipynb) | Answers questions about the Llama 2 paper (77 pages) using only an embedding model, NumPy and an LLM, and compares two chunk sizes by recall. |
| 3. Running a model locally | [`task3_ollama_local.ipynb`](task3_ollama_local.ipynb) | Runs Qwen2.5-3B-Instruct in Ollama as fp16 and as 4-bit, compares speed, memory and answers on the same 10 prompts, and sizes a server for 20 users. |
 
## Run a notebook
 
1. In Colab choose **File → Open notebook → GitHub**, paste this repository's URL and pick the notebook (or download the `.ipynb` and upload it).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.** Roughly 10 minutes for Task 1, a few minutes for Task 2, and 10 to 15 minutes for Task 3 (about 8 GB of model downloads).
4. To show the results on GitHub, download the finished notebook (**File → Download → .ipynb**, outputs included) and commit it in place of the empty one.
## Where each deliverable is
 
Code cells are numbered `# 1` to `# 5` in their first line. "Last section" is the final text cell of the notebook.
 
| Task | Deliverable | Where |
|---|---|---|
| 1 | Base vs fine-tuned accuracy on at least 50 held-out examples | cells 3 and 5 |
| 1 | LoRA settings and why | values in cell 1, reasons in the table before cell 4 |
| 1 | 3 failures and why | cell 5, causes to look for in the last section |
| 1 | 7B memory: full fine-tuning, LoRA, QLoRA | last section |
| 2 | 15 questions with their source passages | cell 2 |
| 2 | Recall for two chunk sizes, which is better and why | cell 3 (numbers), last section (reasons) |
| 2 | 2 retrieval failures and what would fix them | cell 4, last section |
| 2 | BM25 and re-ranker | last section |
| 3 | Tokens per second and memory for each setting | cell 3 |
| 3 | Quality on the same 10 prompts | cells 3 (rough auto-check) and 4 (answers side by side) |
| 3 | KV cache and why quantisation saves memory | cell 5 (numbers), last section |
| 3 | Serving 20 users on one 24 GB GPU | last section |
 
## Task 1: LoRA fine-tuning
 
- **Data and metric.** `dair-ai/emotion`, six labels. 3,000 training tweets; about 300 held-out tweets from the official test split (any text that also appears in training is removed, so the printed count can be a little lower). An answer is correct if it starts with the true label (greedy decoding). The base and fine-tuned models get the identical prompt.
- **LoRA settings.** Rank 16, alpha 32, dropout 0.05, all linear layers (`q, k, v, o, gate, up, down`), learning rate 2e-4 with cosine decay and 20 warm-up steps, 2 epochs, batch size 16. The notebook gives the reason for each.
- **Training.** Plain `transformers.Trainer` and `peft`. The loss covers only the answer tokens (the prompt is masked), and the code asserts this.
- **Memory for a 7B model (rough).** Full fine-tuning ≈ 115 GB, LoRA ≈ 18 GB, QLoRA ≈ 8 GB. This assumes AdamW, LoRA rank 16 on all linear layers, a Llama-7B shape, batch 4 × 512 tokens and gradient checkpointing; the working is in the notebook.
## Task 2: RAG from scratch
 
- **Pipeline.** PyMuPDF extracts the text of the Llama 2 paper (arXiv 2307.09288, 77 pages, about 42,000 words). Words are grouped into chunks with 25 words of overlap, embedded with `BAAI/bge-small-en-v1.5`, and searched by cosine similarity: a NumPy dot product, which is instant at this size, so no FAISS. The top 5 chunks (300 words each by default) go to Qwen2.5-1.5B-Instruct, which answers from them only. No LangChain, LlamaIndex or vector database.
- **Test set.** 15 questions, each with a source passage of at most 25 words copied verbatim from the paper. The code checks that every passage occurs exactly once and on a single page.
- **Metric.** A hit means a retrieved chunk contains the source passage. The notebook reports recall@1 and recall@5 for 100-word and 300-word chunks, plus recall at equal text (top 15 small chunks against top 5 big chunks), because at the same k bigger chunks show the retriever more text.
- **Not implemented.** BM25 and re-ranking are explained in the notebook, not coded; the assignment asks only for the explanation.
- **Another document.** Change `URL` in cell 1 (any public PDF of 30 or more pages) and rewrite the questions in cell 2.
## Task 3: Running a model locally
 
- **Setup.** Ollama installed in Colab; Qwen2.5-3B-Instruct as `fp16` (6.2 GB) and `q4_K_M` (1.9 GB). Same 10 prompts, temperature 0 and 4,096-token context, so the only difference is the quantisation.
- **Measured.** Generated tokens per second (Ollama's own timings, model loading excluded); file size and the memory Ollama holds for the loaded model, with the part on the GPU; a rough auto-check (does the answer contain the expected word or number?); and all answers side by side for judging quality. The KV-cache size per token is computed from the model's own config.
- **20 users on one 24 GB GPU (back-of-envelope).** An 8B-class model needs 128 KiB of KV cache per token. After weights and overhead, fp16 leaves room for about 6 chats of 4,096 tokens and 4-bit for about 26. So 20 users need 4-bit weights and a server with continuous batching and paged KV memory (vLLM), giving each user roughly 17 to 56 tokens/s depending on the card (L4 to RTX 4090). The assumptions are in the notebook.
- **No GPU.** Set `BASE` to `qwen2.5:0.5b-instruct` in cell 2; it runs on CPU, more slowly.
## Troubleshooting
 
- **Task 1, `ImportError ... incompatible version of torchao`** at `get_peft_model`: Colab's preinstalled `torchao` is older than `peft` accepts. Add `!pip uninstall -y -q torchao` below the `!pip install` line in cell 1 and run again.
- **Task 1, training loss prints `nan`:** set `fp16=False` in cell 4 (slower, still fits on a T4).
- **Colab asks for a restart after the first cell:** restart the runtime and run all again.
- **Libraries are not pinned.** If a later release breaks a cell, install the previous version of that library.
 
