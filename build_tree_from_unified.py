# build_tree_from_unified.py
import pandas as pd
import re, unicodedata
from pathlib import Path
from rapidfuzz import process, fuzz

SRC_PATH = "tabela_falhas_unificada.xlsx"   # seu arquivo
SHEET    = "Sheet1"                         # nome exato da aba (troque se necessário)
OUT_PATH = "arvore_decisao.xlsx"

# ---------- util ----------
def norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def split_checks(texto):
    """Divide verificação em passos por ; quebra de linha • - etc."""
    if not str(texto).strip():
        return []
    raw = re.split(r"[;\n\r•\-]+", str(texto))
    return [s.strip() for s in raw if s.strip()]

def pick_col_fuzzy(cols, candidates, min_score=70):
    """
    Escolhe a melhor coluna comparando por similaridade (token_set_ratio).
    Retorna nome real da coluna ou None.
    """
    if not cols:
        return None
    # lista “ampliada” de candidatos (variações comuns)
    expanded = list(dict.fromkeys(candidates + [c.replace("ç","c").replace("ã","a").replace("õ","o") for c in candidates]))
    # melhor match entre todas as colunas e candidatos
    best_name, best_score = None, -1
    for col in cols:
        for cand in expanded:
            score = fuzz.token_set_ratio(norm(col), norm(cand))
            if score > best_score:
                best_name, best_score = col, score
    return best_name if best_score >= min_score else None

# ---------- leitura ----------
base = Path(SRC_PATH)
if not base.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {base.resolve()}")

df = pd.read_excel(SRC_PATH, sheet_name=SHEET).fillna("")
cols = list(df.columns)

print("\n[COLUNAS ENCONTRADAS NA ABA]", SHEET)
for c in cols:
    print(" -", c)

# Alvos a detectar (listas com sinônimos comuns)
CANDS_DESC  = ["Descrição do Problema","Descricao do Problema","descricao problema","Sintoma","Falha","Defeito","Problema","Descrição","Titulo Falha","Titulo"]
CANDS_VERIF = ["Verificação","Como verificar","Checagem","Passos de verificação","Teste","Inspeção","Inspecao","Checklist","Diagnóstico","Diagnostico","Passos"]
CANDS_SOL   = ["Solução","Acao","Ação","Ação corretiva","Correção","Procedimento","Correcoes","Solucao","Q3","Ação Q3"]
CANDS_SOL2  = ["Q4","Observações finais","Observacao","Obs","Complemento","Ações adicionais","Acao adicional"]
CANDS_CAUSA = ["Causa","Causa provável","Possível causa","Causas","Root cause","Motivo"]
CANDS_GROUP = ["Sistema","Subsistema","Equipamento","Linha","Processo","Area","Setor","Grupo"]

# mapeamento fuzzy
COL_DESC  = pick_col_fuzzy(cols, CANDS_DESC, 70)
COL_VERIF = pick_col_fuzzy(cols, CANDS_VERIF, 70)
COL_SOL   = pick_col_fuzzy(cols, CANDS_SOL, 70)
COL_SOL2  = pick_col_fuzzy(cols, CANDS_SOL2, 70)
COL_CAUSA = pick_col_fuzzy(cols, CANDS_CAUSA, 70)
COL_GROUP = pick_col_fuzzy(cols, CANDS_GROUP, 70)

print("\n[MAPEAMENTO SUGERIDO]")
print("Descrição:",  COL_DESC  or "(não encontrada)")
print("Verificação:",COL_VERIF or "(não encontrada)")
print("Solução:",    COL_SOL   or "(não encontrada)")
print("Complemento:",COL_SOL2  or "(não encontrada)")
print("Causa:",      COL_CAUSA or "(não encontrada)")
print("Grupo/Árvore:",COL_GROUP or "(um item por linha)")

if not COL_DESC:
    raise ValueError("Não identifiquei a coluna de descrição/sintoma. Veja as colunas listadas e, se necessário, me diga qual é o nome exato para eu fixar.")

rows = []

def add_tree_rows(tid, desc, verifs, sol, sol2, causa, keywords):
    # start
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
    # verificações
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
    # folha final
    finale = []
    if causa: finale.append(f"Causa: {causa}")
    if sol:   finale.append(f"Solução: {sol}")
    if sol2:  finale.append(f"Complemento: {sol2}")
    if not finale: finale.append("Solução: -")

    rows.append({
        "TREE_ID": tid, "NODE_ID": "leaf_final", "KIND": "LEAF",
        "TEXT": "",
        "OPT1_LABEL": "", "OPT1_NEXT": "",
        "OPT2_LABEL": "", "OPT2_NEXT": "",
        "OPT3_LABEL": "", "OPT3_NEXT": "",
        "OPT4_LABEL": "", "OPT4_NEXT": "",
        "OPT5_LABEL": "", "OPT5_NEXT": "",
        "SOL_Q3": "\n".join(finale), "SOL_Q4": "-", "KEYWORDS": ""
    })
    # folha alternativa
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

# Gera árvores
if COL_GROUP:
    i = 0
    for _, r in df.iterrows():
        i += 1
        tid = make_id(norm(r[COL_GROUP]) or "tree", i)
        desc  = str(r[COL_DESC]).strip()
        verif = split_checks(r.get(COL_VERIF, "")) if COL_VERIF else []
        sol   = str(r.get(COL_SOL, "")).strip()   if COL_SOL   else ""
        sol2  = str(r.get(COL_SOL2, "")).strip()  if COL_SOL2  else ""
        causa = str(r.get(COL_CAUSA, "")).strip() if COL_CAUSA else ""
        kw    = f"{r[COL_GROUP]} {desc}"
        add_tree_rows(tid, desc, verif, sol, sol2, causa, kw)
else:
    i = 0
    for _, r in df.iterrows():
        i += 1
        tid   = make_id("tree", i)
        desc  = str(r[COL_DESC]).strip()
        verif = split_checks(r.get(COL_VERIF, "")) if COL_VERIF else []
        sol   = str(r.get(COL_SOL, "")).strip()   if COL_SOL   else ""
        sol2  = str(r.get(COL_SOL2, "")).strip()  if COL_SOL2  else ""
        causa = str(r.get(COL_CAUSA, "")).strip() if COL_CAUSA else ""
        add_tree_rows(tid, desc, verif, sol, sol2, causa, desc)

nodes = pd.DataFrame(rows)

# ordenação amigável
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

print(f"\n[OK] Árvore gerada em: {Path(OUT_PATH).resolve()}")
