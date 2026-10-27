"""
Retrieval evaluation for ResearchFlow.

Scores how often the retriever finds the page that actually answers a question,
using 32 hand-written questions in eval/questions.json. Each question is labeled
with the PDF page(s) that answer it and is worded differently from the source
text, so a retriever can't win by matching exact phrases.

Retrievers compared, all over the same chunks the app uses (1,000 chars, 200 overlap):
  bm25    keyword search (rank_bm25), no API key needed
  dense   Gemini embeddings, the retriever ResearchFlow uses today
  hybrid  bm25 + dense merged with reciprocal rank fusion (RRF)

Metrics:
  Recall@1, Recall@4  share of questions with a correct page in the top 1 / top 4
  MRR                 mean reciprocal rank of the first correct chunk (in the top 10)
  ms/query            median retrieval time

Run from the repo root:
  python eval/evaluate.py            # all three (needs GOOGLE_API_KEY in .env)
  python eval/evaluate.py --bm25     # keyword baseline only, no key needed
"""

import json
import os
import re
import statistics
import sys
import time

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

K = 4          # what the app retrieves today
DEPTH = 10     # how deep we look for MRR and for fusing rankings
RRF_K = 60     # standard reciprocal rank fusion constant


def load_chunks():
    docs = DirectoryLoader(path="./data/", glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def page_of(chunk):
    return (os.path.basename(chunk.metadata["source"]), chunk.metadata["page"])


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, chunks):
        self.index = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def rank(self, query):
        scores = self.index.get_scores(tokenize(query))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:DEPTH]


class Dense:
    def __init__(self, chunks):
        from dotenv import load_dotenv
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        load_dotenv()
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = i
        self.store = Chroma.from_documents(
            chunks,
            GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"),
            collection_name="eval",
        )

    def rank(self, query):
        hits = self.store.similarity_search(query, k=DEPTH)
        return [h.metadata["chunk_id"] for h in hits]


class Hybrid:
    def __init__(self, bm25, dense):
        self.bm25, self.dense = bm25, dense

    def rank(self, query):
        fused = {}
        for ranking in (self.bm25.rank(query), self.dense.rank(query)):
            for position, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (RRF_K + position + 1)
        return sorted(fused, key=lambda i: -fused[i])[:DEPTH]


def evaluate(name, retriever, chunks, questions):
    hit1 = hitk = 0
    reciprocal_ranks, times, misses = [], [], []
    for item in questions:
        relevant = {tuple(p) for p in item["pages"]}
        start = time.perf_counter()
        ranking = retriever.rank(item["q"])
        times.append((time.perf_counter() - start) * 1000)
        found = [page_of(chunks[i]) in relevant for i in ranking]
        hit1 += found[0]
        hitk += any(found[:K])
        first = found.index(True) + 1 if True in found else None
        reciprocal_ranks.append(1 / first if first else 0)
        if not any(found[:K]):
            misses.append(item["q"])
    n = len(questions)
    print(f"{name:7s} Recall@1 {hit1 / n:6.1%}   Recall@{K} {hitk / n:6.1%}   "
          f"MRR {statistics.mean(reciprocal_ranks):.3f}   "
          f"{statistics.median(times):7.1f} ms/query")
    return misses


def main():
    chunks = load_chunks()
    questions = json.load(open("eval/questions.json"))
    print(f"{len(questions)} questions, {len(chunks)} chunks\n")

    bm25 = BM25(chunks)
    results = {"bm25": evaluate("bm25", bm25, chunks, questions)}
    if "--bm25" not in sys.argv:
        dense = Dense(chunks)
        results["dense"] = evaluate("dense", dense, chunks, questions)
        results["hybrid"] = evaluate("hybrid", Hybrid(bm25, dense), chunks, questions)

    for name, misses in results.items():
        if misses:
            print(f"\n{name} missed in top {K}:")
            for q in misses:
                print(f"  - {q}")


if __name__ == "__main__":
    main()
