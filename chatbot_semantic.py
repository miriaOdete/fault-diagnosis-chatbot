import os
import pandas as pd
import unicodedata, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

class SemanticRouter:
    MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, arvore_engine, excel_q34="arvore_decisao.xlsx", aba="Sheet1"):
        self.engine = arvore_engine
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.excel_q34 = os.path.join(base_dir, excel_q34)
        self.aba = aba
        self._load_q34()
        self.model = SentenceTransformer(self.MODEL)
        self._build_tree_index()

    def _normalize(self, s: str) -> str:
        s = unicodedata.normalize('NFD', str(s))
        s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    def _load_q34(self):
        if not os.path.exists(self.excel_q34):
            self.df_q = None
            return
        df = pd.read_excel(self.excel_q34, sheet_name=self.aba).fillna("")

        def find(cols, cands):
            norm = {self._normalize(c): c for c in cols}
            for c in cands:
                k = self._normalize(c)
                if k in norm:
                    return norm[k]
            return None

        col_desc = find(df.columns, ["Descrição do Problema", "descricao do problema", "descricao_problema"])
        col_q3 = find(df.columns, ["Q3", "q3"])
        col_q4 = find(df.columns, ["Q4", "q4"])

        if not all([col_desc, col_q3, col_q4]):
            self.df_q = None
            return

        self.col_desc, self.col_q3, self.col_q4 = col_desc, col_q3, col_q4
        self.df_q = df

    def _build_tree_index(self):
        self.tree_ids = self.engine.list_trees()
        texts = []
        for tid in self.tree_ids:
            kw = self.engine.keywords.get(tid, "") or ""
            start = self.engine.get_start(tid)
            st = start["text"] if start else ""
            texts.append(f"{kw} {st}".strip())
        self.tree_texts = texts
        self.tree_emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.tree_map = self.tree_ids[:]

    def _hybrid_score(self, query: str, candidates: list[str]):
        if not candidates:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        cand_emb = self.model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb, cand_emb)[0]
        fuzzy_scores = [fuzz.token_set_ratio(query, c) / 100.0 for c in candidates]
        final = [0.7 * s + 0.3 * f for s, f in zip(sims, fuzzy_scores)]
        return list(final), list(sims), list(fuzzy_scores)

    def pick_tree(self, user_problem: str, limiar=0.45):
        if not self.tree_ids:
            return None, 0.0
        finals, sims, fuzzys = self._hybrid_score(user_problem, self.tree_texts)
        best_idx = int(np.argmax(finals))
        best = float(finals[best_idx])
        return (self.tree_map[best_idx] if best >= limiar else None), best

    def fallback_q34(self, user_problem: str):
        if self.df_q is None or len(self.df_q) == 0:
            return "Desculpe, não encontrei nenhuma solução correspondente."

        texts = self.df_q[self.col_desc].astype(str).tolist()
        finals, sims, fuzzys = self._hybrid_score(user_problem, texts)
        idxs = list(np.argsort(finals)[::-1])[:3]
        best_idx = int(idxs[0]) if idxs else -1
        best_val = float(finals[best_idx]) if idxs else 0.0

        LIMIAR = 0.45
        if best_idx < 0 or best_val < LIMIAR:
            sugest = [f"- {texts[i]}" for i in idxs]
            return "Não encontrei uma correspondência exata.\nSugestões próximas:\n" + "\n".join(sugest)

        row = self.df_q.iloc[best_idx]
        desc = str(row[self.col_desc]).strip()
        q3 = str(row[self.col_q3]).strip()
        q4 = str(row[self.col_q4]).strip()

        return "\n".join([
            "Problema encontrado:",
            desc if desc else "-",
            "",
            "Solução Q3:",
            q3 if q3 else "-",
            "",
            "Solução Q4:",
            q4 if q4 else "-",
        ])
