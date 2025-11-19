import re
from typing import List, Dict, Any

import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import pipeline

import time
import csv
from textwrap import shorten


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
rewriter = pipeline("text2text-generation", model="google/flan-t5-small")


def read_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

resume_pdf_path ="Anubhav_Singh_resume.pdf"
jd_pdf_path = "Data_Engineer_JD.pdf"

resume_raw = read_pdf(resume_pdf_path)
jd_raw = read_pdf(jd_pdf_path)

print("Sample of resume text:\n", resume_raw[:500])
print("\n" + "="*80 + "\n")
print("Sample of JD text:\n", jd_raw[:500])


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()
    
def split_into_sentences_or_bullets(text: str) -> List[str]:
    if not text:
        return []
    
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullets = []

    for line in lines:
        if re.match(r"^[-•*\u2022]\s+", line):
            line = re.sub(r"^[-•*\u2022]\s+", "", line)
            bullets.append(line)
        else:
            parts = re.split(r"(?<=[.!?;])\s+", line)
            for p in parts:
                p = p.strip()
                if len(p) > 5:
                    bullets.append(p)

    bullets = [b for b in bullets if len(b) > 10]
    return bullets


resume_text = clean_whitespace(resume_raw)
jd_text = clean_whitespace(jd_raw)

resume_bullets = split_into_sentences_or_bullets(resume_text)
jd_bullets = split_into_sentences_or_bullets(jd_text)

print(f"Found {len(resume_bullets)} resume bullets and {len(jd_bullets)} JD bullets.\n")

print("Sample resume bullets:")
for b in resume_bullets[:5]:
    print("-", b)

print("\nSample JD bullets:")
for b in jd_bullets[:5]:
    print("-", b)


def extract_skills_from_text(text: str, skill_vocab: List[str] = None) -> List[str]:
    if skill_vocab is None:
        skill_vocab = [
            "python", "java", "c++", "c", "sql", "excel", "pandas", "numpy",
            "tensorflow", "pytorch", "react", "node.js", "node", "aws", "docker",
            "kubernetes", "git", "tableau", "power bi", "matlab", "django", "flask"
        ]
    text_low = text.lower()
    found = set()
    for s in skill_vocab:
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_low):
            found.add(s)
    return sorted(found)


resume_skills = extract_skills_from_text(resume_text)
jd_skills = extract_skills_from_text(jd_text)
missing_skills = [s for s in jd_skills if s not in resume_skills]

print("Resume skills:", resume_skills)
print("JD skills    :", jd_skills)
print("Missing skills:", missing_skills)


def get_embeddings(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    emb = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb

def similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    return cosine_similarity(A, B)


def compute_match(resume_bullets: List[str], jd_bullets: List[str]) -> Dict[str, Any]:
    emb_resume = get_embeddings(resume_bullets)
    emb_jd = get_embeddings(jd_bullets)

    sim = similarity_matrix(emb_resume, emb_jd)  # shape: (len(resume), len(jd))
    
    jd_matches = []
    jd_scores = []
    weights = []
    
    for j_idx, jd_line in enumerate(jd_bullets):
        if sim.size == 0:
            best_score = 0.0
            best_idx = None
        else:
            col = sim[:, j_idx]
            best_idx = int(np.argmax(col))
            best_score = float(col[best_idx])
        
        # weight based on wording in JD
        w = 1.0
        if re.search(r"\b(required|must|must have|required experience)\b", jd_line, flags=re.I):
            w = 2.0
        if re.search(r"\b(preferred|nice to have|optional)\b", jd_line, flags=re.I):
            w = 0.8
        
        jd_matches.append({
            "jd_bullet": jd_line,
            "best_resume_bullet": resume_bullets[best_idx] if best_idx is not None and resume_bullets else None,
            "score": best_score,
            "resume_index": best_idx,
            "weight": w
        })
        jd_scores.append(best_score)
        weights.append(w)
    
    if jd_scores:
        overall = float(np.average(jd_scores, weights=weights))
    else:
        overall = 0.0
    
    resume_skills = extract_skills_from_text("\n".join(resume_bullets))
    jd_skills = extract_skills_from_text("\n".join(jd_bullets))
    missing_skills = [s for s in jd_skills if s not in resume_skills]
    
    return {
        "overall_score": round(overall * 100, 2),   # 0–100 scale
        "jd_matches": jd_matches,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "missing_skills": missing_skills
    }


match_report = compute_match(resume_bullets, jd_bullets)

print("Overall match score:", match_report["overall_score"])
print("Missing skills:", match_report["missing_skills"])

print("\nTop 5 JD lines with their best resume matches:")
for item in match_report["jd_matches"][:5]:
    print(f"JD: {item['jd_bullet']}")
    print(f"Best resume bullet: {item['best_resume_bullet']}")
    print(f"Score: {item['score']:.3f}, Weight: {item['weight']}")
    print("-" * 60)


from typing import List, Dict, Any
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import torch

sbert_model_name = "all-MiniLM-L6-v2"
sbert = SentenceTransformer(sbert_model_name)

gen_model_name = "google/flan-t5-small"  
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

mnli_model_name = "roberta-large-mnli" 
mnli_tokenizer = AutoTokenizer.from_pretrained(mnli_model_name)
mnli_model = AutoModelForSequenceClassification.from_pretrained(mnli_model_name)
mnli_label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}  

device = 0 if torch.cuda.is_available() else -1
if torch.cuda.is_available():
    gen_model = gen_model.to("cuda")
    mnli_model = mnli_model.to("cuda")
    sbert = sbert.to("cuda")

mnli_pipe = pipeline("text-classification", model=mnli_model, tokenizer=mnli_tokenizer, device=0 if torch.cuda.is_available() else -1, return_all_scores=True)

def token_set(text: str) -> set:
    return set(re.findall(r"\w+", (text or "").lower()))

def jd_token_overlap_fraction(suggestion: str, jd_text: str) -> float:
    s_tokens = token_set(suggestion)
    jd_tokens = token_set(jd_text)
    if not s_tokens:
        return 0.0
    return len(s_tokens & jd_tokens) / len(s_tokens)

def surface_change_score(original: str, candidate: str) -> float:
    o = token_set(original)
    c = token_set(candidate)
    if not o and not c:
        return 0.0
    overlap = len(o & c) / (len(o | c) + 1e-9)
    return 1.0 - overlap

def semantic_similarity(a: str, b: str) -> float:
    emb = sbert.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    a_emb, b_emb = emb[0], emb[1]
    cos = np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-9)
    return float(np.clip(cos, -1.0, 1.0))

def entailment_probability(premise: str, hypothesis: str) -> float:
    inputs = mnli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        mnli_model.to("cuda")
    with torch.no_grad():
        out = mnli_model(**inputs)
        logits = out.logits.squeeze().cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum()
        entail_prob = float(probs[2])
    return entail_prob

def gen_candidates_t5(original_bullet: str, jd_text: str, num_return: int = 4, max_new_tokens: int = 64) -> List[str]:
    prompt = (
        "Rewrite the following resume bullet into a single concise, professional sentence, "
        "using a stronger action verb and clarifying the impact. DO NOT invent new tools or years; "
        "use ONLY facts present in the original sentence.\n\n"
        f"Original: {original_bullet}\n\n"
        "Rewritten:"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    candidates = []

    out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, num_return_sequences=1, early_stopping=True)
    text = gen_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    candidates.append(text)

   
    for _ in range(max(0, num_return-1)):
        out = gen_model.generate(**inputs, do_sample=True, top_p=0.9, top_k=50, temperature=0.8, max_new_tokens=max_new_tokens, num_return_sequences=1)
        text = gen_tokenizer.decode(out[0], skip_special_tokens=True).strip()
        candidates.append(text)

   
    uniq = []
    for c in candidates:
        c = re.sub(r"^[\-\u2022\*\s0-9\.]+", "", c).split("\n",1)[0].strip()
        if c and c not in uniq:
            uniq.append(c)
    return uniq[:num_return]

def advanced_rewrite_bullet(original_bullet: str, jd_text: str, num_candidates: int = 4,
                            min_semantic_sim: float = 0.55, max_jd_overlap: float = 0.45,
                            min_entailment: float = 0.55, min_surface_change: float = 0.15) -> List[str]:

    if not original_bullet or not original_bullet.strip():
        return []

    candidates = gen_candidates_t5(original_bullet, jd_text, num_return=num_candidates)
    candidates.append(simple_rule_based_rewrite(original_bullet, jd_text=jd_text))
    scored = []
    orig = original_bullet
    for cand in candidates:
        if cand.strip().lower() == orig.strip().lower():
            continue

        jd_overlap = jd_token_overlap_fraction(cand, jd_text)
        if jd_overlap > max_jd_overlap:
            continue

        semsim = semantic_similarity(orig, cand)
        if semsim < min_semantic_sim:
            continue

        surf = surface_change_score(orig, cand)
        if surf < min_surface_change:
            continue

        ent_prob = entailment_probability(orig, cand)
        if ent_prob < min_entailment:
            continue
        score = 0.5 * ent_prob + 0.3 * semsim + 0.2 * surf
        scored.append({"candidate": cand, "score": score, "ent": ent_prob, "sem": semsim, "surf": surf, "jd_overlap": jd_overlap})

    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)

    if not scored_sorted:
        fallback = simple_rule_based_rewrite(original_bullet, jd_text=jd_text)
        return [fallback]

    results = []
    seen = set()
    for item in scored_sorted:
        c = item["candidate"].strip()
        low = c.lower()
        if low not in seen:
            results.append(c)
            seen.add(low)
        if len(results) >= 2:
            break

    return results


MAX_ITEMS = 50         
LOW_SCORE_THRESHOLD = 0.55
PRINT_JD_SNIPPET_LEN = 200
SAVE_TO_CSV = True
CSV_PATH = "rewrites.csv"

def print_rewrites_for_test():
    test_bullet = "Developed a machine learning model in Python to predict customer churn."
    test_jd = "We are looking for a Data Scientist with experience in Python, machine learning, and predictive modeling."
    print("=== Single TEST example ===")
    print("Original:", test_bullet)
    t0 = time.time()
    try:
        res = advanced_rewrite_bullet(test_bullet, test_jd, num_candidates=5)
    except Exception as e:
        print("advanced_rewrite_bullet failed:", e)
        return
    dt = time.time() - t0
    print(f"Generated {len(res)} candidate(s) in {dt:.2f}s:")
    for r in res:
        print(" -", r)
    print("="*80, "\n")

def print_rewrites_from_match_report(match_report, jd_text=None, max_items=MAX_ITEMS, save_csv=SAVE_TO_CSV):
    if not match_report or 'jd_matches' not in match_report:
        print("No match_report['jd_matches'] found. Please ensure match_report exists.")
        return

    rows = []
    count = 0
    print("=== Batch rewrites from match_report ===\n")
    for idx, item in enumerate(match_report['jd_matches']):
        if count >= max_items:
            break
        score = item.get('score', 0.0)
        if score >= LOW_SCORE_THRESHOLD:
            continue  

        jd_bullet = item.get('jd_bullet', '') or ''
        orig = item.get('best_resume_bullet') or ''
        if not orig:
            continue

        context_jd = jd_text if jd_text else jd_bullet

        try:
            t0 = time.time()
            candidates = advanced_rewrite_bullet(orig, context_jd, num_candidates=5)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"[IDX {idx}] advanced_rewrite_bullet failed: {e}")
            continue

        print(f"IDX {idx}  | score={score:.3f}")
        print("JD (snippet):", shorten(jd_bullet, width=PRINT_JD_SNIPPET_LEN, placeholder="..."))
        print("Original resume bullet:", shorten(orig, width=200, placeholder="..."))
        print(f"Candidates ({len(candidates)}) [took {elapsed:.2f}s]:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c}")
        print("-"*80)

        rows.append({
            "idx": idx,
            "score": score,
            "jd_bullet": jd_bullet,
            "original_bullet": orig,
            "candidates": " ||| ".join(candidates)
        })

        count += 1

    print(f"\nProcessed {count} JD items (low-score threshold {LOW_SCORE_THRESHOLD}).")

    if save_csv and rows:
        try:
            with open(CSV_PATH, "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["idx","score","jd_bullet","original_bullet","candidates"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Saved results to {CSV_PATH}")
        except Exception as e:
            print("Failed to save CSV:", e)


if 'match_report' in globals():
    print_rewrites_from_match_report(match_report, jd_text=globals().get('jd_text', None), max_items=MAX_ITEMS, save_csv=SAVE_TO_CSV)
else:
    print("match_report not found in globals(), skipping batch run.")


