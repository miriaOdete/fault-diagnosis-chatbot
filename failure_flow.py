import os
import re
import unicodedata
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
from rapidfuzz import fuzz, process, distance

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

PT_STOPWORDS = {
    "a","o","as","os","um","uma","de","do","da","dos","das","no","na","nos","nas",
    "em","para","por","com","sem","sobre","entre","e","ou","mas","que","se","ao",
    "à","às","aos","é","ser","estar","ter","haver","foi","está","são","tá","tem",
    "têm","já","não","sim","há","quando","onde","como","qual","quais","porque",
    "isso","isto","aquilo","essa","esse","devido","pela","pelas","pelo","pelos"
}

def _normalize(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _tokens(text: str) -> List[str]:
    t = _normalize(text)
    return [w for w in t.split() if len(w) >= 3 and w not in PT_STOPWORDS]

PATTERNS = [
    (r"\bfalta de (.+)", lambda g: f"Você percebeu falta de {g}?"),
    (r"\bindisponibilidade de (.+)", lambda g: f"Há indisponibilidade de {g}?"),
    (r"\bsensor(es)? de ([\w\s]+) (defeituoso|com falha|com problema|instável|instavel)",
     lambda g1, g2, g3: f"O sensor de {g2} está {g3.replace('instavel','instável')}?"),
    (r"\bsensor(es)? (defeituoso|com falha|com problema|instável|instavel)",
     lambda g1, g2: f"Algum sensor está {g2.replace('instavel','instável')}?"),
    (r"\bvazamento(s)?( de)? ([\w\s]+)?", lambda *_g: "Existe vazamento (óleo, ar, fluido) no ponto afetado?"),
    (r"\b(entupimento|obstrucao|obstrução) (do|da|de)? ([\w\s]+)?",
     lambda *_g: "O componente ou linha apresenta entupimento/obstrução?"),
    (r"\b(desalinhamento|alinhamento incorreto)([\w\s]*)",
     lambda *_g: "Há desalinhamento aparente entre os componentes?"),
    (r"\b(folga|folgas?) (excessiva|excessivas)?([\w\s]*)",
     lambda *_g: "Existe folga excessiva na peça ou conjunto?"),
    (r"\b(oxidacao|oxidação|corrosao|corrosão)([\w\s]*)",
     lambda *_g: "Há sinais de oxidação/corrosão nas partes afetadas?"),
    (r"\b(quebra|trinca|trincas|trincado|fratura)([\w\s]*)",
     lambda *_g: "Alguma peça está trincada ou quebrada?"),
    (r"\b(superaquecimento|temperatura elevada|temperatura alta)([\w\s]*)",
     lambda *_g: "A temperatura do conjunto está acima do normal?"),
    (r"\b(baixa pressao|pressao baixa|alta pressao|pressao alta)([\w\s]*)",
     lambda *_g: "A leitura de pressão está fora do especificado?"),
    (r"\b(cabo|conector|fiação|fios?)([\w\s]*) (solto|danificado|quebrado|com folga)",
     lambda *_g: "Cabos ou conectores estão soltos ou danificados?"),
    (r"\b(contaminacao|contaminação|sujeira|impurezas)([\w\s]*)",
     lambda *_g: "Foi observada contaminação ou sujeira no sistema?"),
    (r"\b(desgaste|desgastado|desgastes)([\w\s]*)",
     lambda *_g: "Há desgaste aparente no componente?"),
    (r"\b(software|configuracao|configuração|parametrizacao|parametrização)([\w\s]*)",
     lambda *_g: "Pode haver erro de software ou configuração incorreta?"),
    (r"\b(alimentacao|alimentação|energia|tensao|tensão)([\w\s]*)",
     lambda *_g: "Há problema de alimentação/energia?"),
    (r"\b(lubrificacao|lubrificação) (insuficiente|inadequada|ausente)",
     lambda *_g: "A lubrificação está ausente ou insuficiente?")
]

def _clean_for_question(txt: str) -> str:
    t = str(txt).strip().rstrip(".;,:- ")
    if not t: return ""
    return t[0].upper() + t[1:]

def cause_to_question(cause_text: str) -> str:
    raw = str(cause_text).strip()
    norm = _normalize(raw)
    for pat, builder in PATTERNS:
        m = re.search(pat, norm)
        if m:
            try:
                q = builder(*m.groups())
            except TypeError:
                g = [g for g in (m.groups() or []) if g]
                q = builder(*g) if g else None
            if q:
                return _clean_for_question(q if q.endswith("?") else q + "?")
    base = _clean_for_question(raw)
    if base.lower().startswith(("o ","a ","os ","as ")):
        q = base + "?"
    else:
        q = f"Você observou {base[0].lower() + base[1:]}?"
    return _clean_for_question(q)

class FailureFlowEngine:
    def __init__(self, excel_path="tabela_falhas_unificada.xlsx", sheet="Sheet1",
                 model_dir="models", enable_spellcheck=True, custom_vocab_path="custom_vocab.txt"):
        self.excel_path = excel_path
        self.sheet = sheet
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "modes_tfidf.joblib")
        self.base_csv = os.path.join(model_dir, "base_unificada.csv")

        self.col_mode = None
        self.col_effect = None
        self.col_cause = None
        self.col_action = None

        self.enable_spellcheck = enable_spellcheck and (SpellChecker is not None)
        self.custom_vocab_path = custom_vocab_path
        self.spell = None
        self.domain_vocab = set()
        self.user_vocab = set()

        if os.path.exists(self.model_path) and os.path.exists(self.base_csv):
            self._load()
        else:
            self._fit_from_excel()

        if self.enable_spellcheck:
            try:
                self.spell = SpellChecker(language="pt")
            except Exception:
                self.spell = None
                self.enable_spellcheck = False
            self._build_domain_vocab_selective()
            self._load_user_vocab()

    def _split_words(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", str(text))

    def _build_domain_vocab_selective(self, min_freq=3):
        freq = {}
        cols = [c for c in [self.col_mode, self.col_effect, self.col_cause, self.col_action] if c]
        for c in cols:
            for val in self.df[c].astype(str).tolist():
                for w in self._split_words(val):
                    freq[w] = freq.get(w, 0) + 1
        wl = set()
        for w, n in freq.items():
            if w.isupper() or any(ch.isdigit() for ch in w) or "-" in w or "_" in w or n >= min_freq or len(w) <= 2:
                wl.add(w)
                wl.add(w.lower())
        self.domain_vocab = wl

    def _load_user_vocab(self):
        if self.custom_vocab_path and os.path.exists(self.custom_vocab_path):
            try:
                with open(self.custom_vocab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            self.user_vocab.add(w)
                            self.user_vocab.add(w.lower())
            except Exception:
                pass

    def _pick_col_fuzzy(self, cols: List[str], candidates: List[str], min_score=70):
        best_name, best_score = None, -1
        for col in cols:
            for cand in candidates:
                score = fuzz.token_set_ratio(_normalize(col), _normalize(cand))
                if score > best_score:
                    best_name, best_score = col, score
        return best_name if best_score >= min_score else None

    def _fit_from_excel(self):
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet).fillna("")
        cols = list(df.columns)

        C_MODE = ["Modos de Falha","Modo de Falha","Modo Falha","Failure Mode","Modo"]
        C_EFFECT = ["Efeitos de Falha","Efeito de Falha","Efeito","Failure Effect","Consequência"]
        C_CAUSE  = ["Causa da Falha","Causas da Falha","Causa","Root Cause","Causa Provável","Possível Causa"]
        C_ACTION = ["Ação Preventiva Recomendada/Troubleshooting","Ação Preventiva","Troubleshooting",
                    "Ações Recomendadas","Ação Recomendada","Acoes Preventivas"]

        self.col_mode = self._pick_col_fuzzy(cols, C_MODE) or cols[0]
        self.col_effect = self._pick_col_fuzzy(cols, C_EFFECT) or cols[1 if len(cols)>1 else 0]
        self.col_cause = self._pick_col_fuzzy(cols, C_CAUSE)
        self.col_action = self._pick_col_fuzzy(cols, C_ACTION)

        self.df = df
        self.df["_mode_n"] = df[self.col_mode].astype(str).map(_normalize)
        self.df["_effect_n"] = df[self.col_effect].astype(str).map(_normalize)

        self.modes_unique = sorted(df[self.col_mode].astype(str).dropna().unique(), key=lambda x: _normalize(x))

        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), strip_accents="unicode")
        X = self.vectorizer.fit_transform(self.modes_unique)
        self.nn = NearestNeighbors(n_neighbors=min(5, len(self.modes_unique)), metric="cosine").fit(X)

        self.mode_effects = {}
        for mode in self.modes_unique:
            sel = self.df[self.df["_mode_n"] == _normalize(mode)]
            effs = sorted(sel[self.col_effect].astype(str).dropna().unique(), key=lambda x: _normalize(x))
            self.mode_effects[mode] = effs

        dump({
            "vectorizer": self.vectorizer,
            "nn": self.nn,
            "modes_unique": self.modes_unique,
            "col_mode": self.col_mode,
            "col_effect": self.col_effect,
            "col_cause": self.col_cause,
            "col_action": self.col_action
        }, self.model_path)

        df.to_csv(self.base_csv, index=False, encoding="utf-8")

    def _load(self):
        store = load(self.model_path)
        self.vectorizer = store["vectorizer"]
        self.nn = store["nn"]
        self.modes_unique = list(store["modes_unique"])
        self.col_mode = store["col_mode"]
        self.col_effect = store["col_effect"]
        self.col_cause = store["col_cause"]
        self.col_action = store["col_action"]

        self.df = pd.read_csv(self.base_csv).fillna("")
        self.df["_mode_n"] = self.df[self.col_mode].astype(str).map(_normalize)
        self.df["_effect_n"] = self.df[self.col_effect].astype(str).map(_normalize)

        self.mode_effects = {}
        for mode in self.modes_unique:
            sel = self.df[self.df["_mode_n"] == _normalize(mode)]
            effs = sorted(sel[self.col_effect].astype(str).dropna().unique(), key=lambda x: _normalize(x))
            self.mode_effects[mode] = effs

    def retrain(self):
        self._fit_from_excel()
        if self.enable_spellcheck:
            self._build_domain_vocab_selective()
            self._load_user_vocab()

    def pick_mode(self, user_text: str, limiar_ok=0.45):
        q = self.vectorizer.transform([user_text])
        dist, idx = self.nn.kneighbors(q, n_neighbors=min(3, self.nn.n_neighbors))
        sims = 1 - dist[0]
        ids = idx[0]
        suggestions = [self.modes_unique[i] for i in ids]
        if not suggestions:
            return None, 0.0, []
        if sims[0] >= limiar_ok:
            return suggestions[0], float(sims[0]), suggestions
        return None, float(sims[0]), suggestions

    def list_effects(self, mode: str):
        return list(self.mode_effects.get(mode, []))

    def choose_from(self, text: str, options: List[str]):
        if not options:
            return None
        best = process.extractOne(text, options, scorer=fuzz.token_set_ratio)
        return best[0] if best and best[1] >= 60 else None

    def _causes_for(self, mode: str, effect: str):
        sel = self.df[(self.df["_mode_n"] == _normalize(mode)) &
                      (self.df["_effect_n"] == _normalize(effect))]
        if sel.empty:
            sel = self.df[self.df["_mode_n"] == _normalize(mode)]
        if not self.col_cause:
            return []
        causas = [c for c in sel[self.col_cause].astype(str).tolist() if str(c).strip()]
        seen, out = set(), []
        for c in causas:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _actions_for_causes(self, mode: str, effect: str, chosen_causes: List[str]):
        if not self.col_action:
            return []
        sel = self.df[(self.df["_mode_n"] == _normalize(mode)) &
                      (self.df["_effect_n"] == _normalize(effect))]
        if sel.empty:
            sel = self.df[self.df["_mode_n"] == _normalize(mode)]

        actions = []
        cause_norms = [_normalize(c) for c in chosen_causes]
        for _, row in sel.iterrows():
            cause_txt = str(row.get(self.col_cause, "")).strip()
            act_txt = str(row.get(self.col_action, "")).strip()
            if not act_txt:
                continue
            if not cause_txt:
                actions.append(act_txt)
                continue
            if any(_normalize(cause_txt) == cn or cn in _normalize(cause_txt) for cn in cause_norms):
                actions.append(act_txt)

        seen, out = set(), []
        for a in actions:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def _should_skip_token(self, w: str):
        if not w or len(w) <= 2:
            return True
        if any(ch.isdigit() for ch in w):
            return True
        if w.isupper():
            return True
        if "-" in w or "/" in w or "_" in w:
            return True
        return False

    def _plausible(self, original, suggestion):
        if not suggestion:
            return False
        if original.lower() == suggestion.lower():
            return False
        return distance.Levenshtein.distance(original.lower(), suggestion.lower()) <= 2

    def _correct_sentence(self, text: str):
        if not self.enable_spellcheck or self.spell is None:
            return text

        parts = re.findall(r"\w+|[^\w\s]", str(text))
        out = []
        for p in parts:
            if re.fullmatch(r"[^\w\s]", p):
                out.append(p)
                continue
            w = p
            if (w in self.user_vocab or w.lower() in self.user_vocab or
                w in self.domain_vocab or w.lower() in self.domain_vocab):
                out.append(w)
                continue
            if self._should_skip_token(w):
                out.append(w)
                continue
            try:
                if w.lower() in self.spell:
                    out.append(w)
                    continue
                suggestion = self.spell.correction(w.lower())
                if suggestion and self._plausible(w, suggestion):
                    if w[0].isupper():
                        suggestion = suggestion.capitalize()
                    out.append(suggestion)
                else:
                    out.append(w)
            except Exception:
                out.append(w)

        result = ""
        for i, t in enumerate(out):
            if i == 0:
                result += t
            else:
                if re.fullmatch(r"[^\w\s]", t):
                    result += t
                else:
                    if result and not result.endswith((" ", "\n")):
                        result += " "
                    result += t
        return result

    def _correct_list(self, lines: List[str]):
        return [self._correct_sentence(s) for s in lines]

    def build_questions(self, mode: str, effect: str, max_q=5):
        causes = self._causes_for(mode, effect)
        if not causes:
            return {"mode": mode, "effect": effect, "causes": [], "questions": [], "scores": [], "q_index": 0}

        questions = []
        used_texts = set()
        for cause in causes:
            qtext = cause_to_question(cause)
            if not qtext:
                continue
            key = _normalize(qtext)
            if key in used_texts:
                continue
            used_texts.add(key)
            questions.append({"text": qtext, "covers": [len(questions)]})

        limited = questions[:max_q]
        for i, q in enumerate(limited):
            q["covers"] = [i]

        if not limited:
            limited = [{"text": cause_to_question(causes[0]), "covers": [0]}]

        for q in limited:
            q["text"] = self._correct_sentence(q["text"])

        return {
            "mode": mode,
            "effect": effect,
            "causes": causes[:len(limited)],
            "questions": limited,
            "scores": [0] * len(limited),
            "q_index": 0
        }

    def answer_question(self, diag_state: Dict[str, Any], user_answer: str):
        if not diag_state.get("questions"):
            return diag_state

        ans = (user_answer or "").strip().lower()
        val = 1 if ans in ("sim","s","yes","y","ok") else 0 if ans in ("nao","não","n","no") else None
        qidx = diag_state["q_index"]

        if qidx < len(diag_state["questions"]) and val is not None:
            covers = diag_state["questions"][qidx]["covers"]
            for i in covers:
                diag_state["scores"][i] += 2 if val == 1 else -1

        diag_state["q_index"] = min(qidx + 1, len(diag_state["questions"]))
        return diag_state

    def finalize(self, diag_state: Dict[str, Any], top_k=3):
        causes = diag_state.get("causes", [])[:]
        scores = diag_state.get("scores", [])[:]

        if not causes:
            return "Não encontrei causas cadastradas para esse cenário."

        idxs = sorted(range(len(causes)), key=lambda i: scores[i], reverse=True)
        top_idxs = idxs[:top_k]
        chosen_causes = [causes[i] for i in top_idxs if scores[i] >= 0] or [causes[idxs[0]]]

        mode = diag_state.get("mode", "")
        effect = diag_state.get("effect", "")
        actions = self._actions_for_causes(mode, effect, chosen_causes)

        if self.enable_spellcheck and self.spell is not None:
            chosen_causes = self._correct_list(chosen_causes)
            actions = self._correct_list(actions)

        linhas = ["✅ Diagnóstico concluído", "", "🔹 Causa(s) provável(is):"]
        for c in chosen_causes:
            linhas.append(f"- {c}")

        linhas += ["", "🔹 Ação(ões) recomendada(s):"]
        if actions:
            for a in actions:
                linhas.append(f"- {a}")
        else:
            linhas.append("- (não cadastrada)")

        return "\n".join(linhas)
