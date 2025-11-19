import pandas as pd
import re, unicodedata
from pathlib import Path
from rapidfuzz import process, fuzz

SRC_PATH = "tabela_falhas_unificada.xlsx"
SHEET = "Sheet1"
OUT_PATH = "arvore_decisao.xlsx"

def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def split_checks(texto):
    if not str(texto).strip():
        return []
    raw = re.split(r"[;\n\r•\-]+", str(texto))
    return [s.strip() for s in raw if s.strip()]

def pick_col_fuzzy(cols, candidates, min_score=70):
    expanded = list(dict.fromkeys(candidates + [c.replace("ç","c").replace("ã","a").replace("õ","o") for c in candidates]))
    best_name, best_score = None, -1
    for col in cols:
        for cand in expanded:
            score = fuzz.token_set_ratio(norm(col), norm(cand))
            if score > best_score:
                best_name, best_score = col, score
    return best_name if best_score >= min_score else None

base = Path(SRC_PATH)
if not base.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {base.resolve()}")

df = pd.read_excel(SRC_PATH, sheet_name=SHEET).fillna("")
cols = list(df.columns)

CANDS_DESC  = ["Descrição do Problema","Descricao do Problema","descricao problema","Sintoma","Falha","Defeito","Problema","Descrição","Titulo Falha","Titulo"]
CANDS_VERIF = ["Verificação","Como verificar","Checagem","Passos de verificação","Teste","Inspeção","Inspecao","Checklist","Diagnóstico","Diagnostico","Passos"]
CANDS_SOL   = ["Solução","Acao","Ação","Ação corretiva","Correção","Procedimento","Correcoes","Solucao","Q3","Ação Q3"]
CANDS_SOL2  = ["Q4","Observações finais","Observacao","Obs","Complemento","Ações adicionais","Acao adicional"]
CANDS_CAUSA = ["Causa","Causa provável","Possível causa","Causas","Root cause","Motivo"]
CANDS_GROUP = ["Sistema","Subsistema","Equipamento","Linha","Processo","Area","Setor","Grupo"]

COL_DESC  = pick_col_fuzzy(cols, CANDS_DESC, 70)
COL_VERIF = pick_col_fuzzy(cols, CANDS_VERIF, 70)
COL_SOL   = pick_col_fuzzy(cols, CANDS_SOL, 70)
COL_SOL2  = pick_col_fuzzy(cols, CANDS_SOL2, 70)
COL_CAUSA = pick_col_fuzzy(cols, CANDS_CAUSA, 70)
COL_GROUP = pick_col_fuzzy(cols, CANDS_GROUP, 70)

if not COL_DESC:
    raise ValueError("Não identifiquei a coluna de descrição/sintoma.")

rows = []

def add_tree_rows(tid, desc, verifs, sol, sol2, causa, keywords):
    rows.append({
        "TREE_ID": tid, "NODE_ID": "start", "KIND": "QUESTION",
        "TEXT": f"O problema observado corresponde a: '{desc}'?",
        "OPT1_LABEL": "Sim", "OPT1_NEXT": "chk_1" if verifs else "leaf_final",
        "OPT2_LABEL": "Não", "OPT2_NEXT": "leaf_alt",
        "OPT3_LABEL": "", "OPT3_NEXT": "",
        "OPT4_LABEL": "", "OPT4_NEXT": "",
        "OPT5_LABEL": "", "OPT5_NEXT": "",
        "SOL_Q3": "", "SOL_Q4": "", "KEYWORDS": keywords or desc
    })
    for i, step in enumerate(verifs, start=1):
        nid = f"chk_{i}"
        nxt = f"chk_{i+1}" if i < len(verifs) else "leaf_final"
        rows.append({
            "TREE_ID": tid, "NODE_ID": nid, "KIND": "QUESTION",
            "TEXT": step,
            "OPT1_LABEL": "Sim", "OPT1_NEXT": nxt,
            "OPT2_LABEL": "Não", "OPT2_NEXT": "leaf_alt",
            "OPT3_LABEL": "", "OPT3_NEXT": "",
            "OPT4_LABEL": "", "OPT4_NEXT": "",
            "OPT5_LABEL": "", "OPT5_NEXT": "",
            "SOL_Q3": "", "SOL_Q4": "", "KEYWORDS": ""
        })
    finale = []
    if causa: finale.append(f"Causa: {causa}")
    if sol: finale.append(f"Solução: {sol}")
    if sol2: finale.append(f"Complemento: {sol2}")
    if not finale: finale.append("Solução: -")

    rows.append({
        "TREE_ID": tid, "NODE_ID": "leaf_final", "KIND": "LEAF",
        "TEXT": "", "OPT1_LABEL": "", "OPT1_NEXT": "",
        "OPT2_LABEL": "", "OPT2_NEXT": "",
        "OPT3_LABEL": "", "OPT3_NEXT": "",
        "OPT4_LABEL": "", "OPT4_NEXT": "",
        "OPT5_LABEL": "", "OPT5_NEXT": "",
        "SOL_Q3": "\n".join(finale), "SOL_Q4": "-", "KEYWORDS": ""
    })
    rows.append({
        "TREE_ID": tid, "NODE_ID": "leaf_alt", "KIND": "LEAF",
        "TEXT": "Não correspondeu às verificações. Revise sintomas e tente outra árvore/sintoma.",
        "OPT1_LABEL": "", "OPT1_NEXT": "",
        "OPT2_LABEL": "", "OPT2_NEXT": "",
        "OPT3_LABEL": "", "OPT3_NEXT": "",
        "OPT4_LABEL": "", "OPT4_NEXT": "",
        "OPT5_LABEL": "", "OPT5_NEXT": "",
        "SOL_Q3": "-", "SOL_Q4": "-", "KEYWORDS": ""
    })

def make_id(prefix, i):
    return f"{prefix}_{i:05d}"

if COL_GROUP:
    i = 0
    for _, r in df.iterrows():
        i += 1
        tid = make_id(norm(r[COL_GROUP]) or "tree", i)
        desc  = str(r[COL_DESC]).strip()
        verif = split_checks(r.get(COL_VERIF, "")) if COL_VERIF else []
        sol   = str(r.get(COL_SOL, "")).strip() if COL_SOL else ""
        sol2  = str(r.get(COL_SOL2, "")).strip() if COL_SOL2 else ""
        causa = str(r.get(COL_CAUSA, "")).strip() if COL_CAUSA else ""
        kw    = f"{r[COL_GROUP]} {desc}"
        add_tree_rows(tid, desc, verif, sol, sol2, causa, kw)
else:
    i = 0
    for _, r in df.iterrows():
        i += 1
        tid = make_id("tree", i)
        desc  = str(r[COL_DESC]).strip()
        verif = split_checks(r.get(COL_VERIF, "")) if COL_VERIF else []
        sol   = str(r.get(COL_SOL, "")).strip() if COL_SOL else ""
        sol2  = str(r.get(COL_SOL2, "")).strip() if COL_SOL2 else ""
        causa = str(r.get(COL_CAUSA, "")).strip() if COL_CAUSA else ""
        add_tree_rows(tid, desc, verif, sol, sol2, causa, desc)

nodes = pd.DataFrame(rows)

def ord_key(nid):
    if nid == "start": return 0
    if str(nid).startswith("chk_"):
        try: return 1 + int(str(nid).split("_",1)[1])
        except: return 50
    if nid == "leaf_final": return 100
    if nid == "leaf_alt": return 101
    return 200

nodes["__ord__"] = nodes["NODE_ID"].map(ord_key)
nodes = nodes.sort_values(["TREE_ID","__ord__"]).drop(columns="__ord__").reset_index(drop=True)

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as w:
    nodes.to_excel(w, sheet_name="Nodes", index=False)
