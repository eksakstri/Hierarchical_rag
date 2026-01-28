from pypdf import PdfReader
import re
import os
import uuid
from typing import List
import numpy as np
import nltk
nltk.download('punkt')
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device="cpu"
)

papers = {}
paper_atomic_units = {}
paper_atomic_embeddings = {}
node_store = {}
paper_roots = {}

def new_id():
    return str(uuid.uuid4())

def cosine(a, b):
    return float(np.dot(a, b))

def to_sentences(text: str) -> List[str]:
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip())>10]

def split_into_atomic_units(text: str):
    parts = re.split(r'(?<=[\.\!\?\;\:])\s+', text)

    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) < 30 and cleaned:
            cleaned[-1] += " " + p
        else:
            cleaned.append(p)

    cleaned = [x for x in cleaned if len(x.strip()) > 0]
    return cleaned

def cluster_semantically(embs, threshold=0.20):
    sim_matrix = np.matmul(embs, embs.T)
    dist = 1 - sim_matrix

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        linkage='average',
        distance_threshold=threshold
    )

    labels = clusterer.fit_predict(dist)
    return labels

def embed_texts(texts, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        arr = encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embs.append(arr)
    return np.vstack(embs)

def embed_query(query):
    return encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

def build_tree_recursive(
        texts, embeddings, paper_name,
        parent_id=None, level=0, max_depth=3, threshold=0.20
    ):

    node_id = new_id()
    combined_text = " ".join(texts)
    node_embedding = encoder.encode(
        [combined_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    node_store[node_id] = {
        "node_id": node_id,
        "parent_id": parent_id,
        "paper_name": paper_name,
        "level": level,
        "text": combined_text,
        "embedding": node_embedding,
        "children": []
    }

    if level >= max_depth or len(texts) <= 2:
        return node_id

    labels = cluster_semantically(embeddings, threshold=threshold)

    cluster_map = {}
    for t, e, l in zip(texts, embeddings, labels):
        cluster_map.setdefault(l, {"texts": [], "embs": []})
        cluster_map[l]["texts"].append(t)
        cluster_map[l]["embs"].append(e)

    for cluster in cluster_map.values():
        child_id = build_tree_recursive(
            cluster["texts"],
            np.vstack(cluster["embs"]),
            paper_name,
            parent_id=node_id,
            level=level + 1,
            max_depth=max_depth,
            threshold=threshold
        )
        node_store[node_id]["children"].append(child_id)

    return node_id
pdfs_path = r"transformer"

for file in os.listdir(pdfs_path):
  if file.endswith(".pdf"):
    pdf_path = os.path.join(pdfs_path, file)
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\n|\r|\f)\s*\d{1,3}\s*(\n|\r|\f)', '\n', text)
    text = text.replace("\x0c", " ")
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s*-\s*', '-', text)
    text = text.replace("•", "-")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    papers[file] = text[:50000]
    print(file)
    print(text[:50000])
    print()

for name, text in papers.items():
    units = split_into_atomic_units(text)
    paper_atomic_units[name] = units
for name, units in paper_atomic_units.items():
    embs = embed_texts(units)
    paper_atomic_embeddings[name] = {
        "units": units,
        "embs": embs
    }
for paper_name, data in paper_atomic_embeddings.items():
    units = data["units"]
    embs = data["embs"]
    root = build_tree_recursive(units, embs, paper_name)
    paper_roots[paper_name] = root

def traverse_node(node_id, query_emb, threshold=0.25):
    node = node_store[node_id]
    node_emb = node["embedding"]
    parent_score = cosine(query_emb, node_emb)

    if parent_score < threshold:
        return []

    if len(node["children"]) == 0:
        return [(node_id, parent_score)]

    child_results = []
    child_scores = []

    for child_id in node["children"]:
        res = traverse_node(child_id, query_emb, threshold)
        if res:
            child_results.extend(res)
            max_child_score = max([score for _, score in res])
            child_scores.append(max_child_score)
        else:
            child_scores.append(0)

    if len(child_scores) == 0 or max(child_scores) < parent_score:
        return [(node_id, parent_score)]

    return child_results

def hierarchical_rag_search(query, threshold=0.25):
    query_emb = embed_query(query)
    results = []

    for paper_name, root_id in paper_roots.items():
        nodes = traverse_node(root_id, query_emb, threshold)
        results.extend(nodes)

    results.sort(key=lambda x: x[1], reverse=True)

    return results

def gather_context(nodes, max_chars=8000):
    weighted_chunks = []

    for rank, (node_id, score) in enumerate(nodes, start=1):
        text = node_store[node_id]["text"]

        weight = (score ** 1.5) / (rank ** 0.75)

        weighted_chunks.append((weight, text))

    weighted_chunks.sort(key=lambda x: x[0], reverse=True)

    combined = ""
    for weight, chunk in weighted_chunks:
        if len(combined) + len(chunk) <= max_chars:
            combined += "\n" + chunk

    return combined.strip()

query = "Explain probabilty"

nodes = hierarchical_rag_search(query, threshold=0.30)

context = gather_context(nodes)

print("Retrieved nodes:", nodes)
print("Final context for LLM:", context[:500])
