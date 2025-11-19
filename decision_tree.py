import pandas as pd
import os
from rapidfuzz import fuzz, process

class DecisionTreeEngine:
    """
    Carrega árvores de decisão do Excel (aba 'Nodes') e navega entre nós.
    Colunas esperadas: TREE_ID, NODE_ID, KIND(QUESTION|LEAF), TEXT,
      OPT1_LABEL/OPT1_NEXT ... OPT5_LABEL/OPT5_NEXT, SOL_Q3, SOL_Q4, KEYWORDS
    """
    def __init__(self, excel_path="arvore_decisao.xlsx", sheet="Nodes"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), excel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo de árvore não encontrado: {path}")

        # Evita dtype misto e o FutureWarning do pandas
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
        df = df.fillna("")
        self.df = df

        self.trees = {}       # TREE_ID -> {node_id -> node_dict}
        self.keywords = {}    # TREE_ID -> keywords string
        self._index()

    def _index(self):
        for tree_id in self.df["TREE_ID"].astype(str).unique():
            sub = self.df[self.df["TREE_ID"].astype(str) == str(tree_id)]
            nodes = {}
            kwords = ""
            for _, r in sub.iterrows():
                node_id = str(r["NODE_ID"]).strip()
                node = {
                    "tree": str(tree_id),
                    "id": node_id,
                    "kind": str(r["KIND"]).strip().upper(),
                    "text": str(r.get("TEXT", "")).strip(),
                    "sol_q3": str(r.get("SOL_Q3", "")).strip(),
                    "sol_q4": str(r.get("SOL_Q4", "")).strip(),
                    "options": []
                }
                # opções OPT1..OPT5
                for i in range(1, 6):
                    lab = str(r.get(f"OPT{i}_LABEL", "")).strip()
                    nxt = str(r.get(f"OPT{i}_NEXT", "")).strip()
                    if lab or nxt:
                        node["options"].append({"label": lab, "next": nxt})
                if not kwords and str(r.get("KEYWORDS", "")).strip():
                    kwords = str(r["KEYWORDS"]).strip()
                nodes[node_id] = node

            self.trees[str(tree_id)] = nodes
            self.keywords[str(tree_id)] = kwords

    # --- API pública usada no app ---
    def list_trees(self):
        return list(self.trees.keys())

    def get_node(self, tree_id, node_id):
        return (self.trees.get(tree_id) or {}).get(node_id)

    def get_start(self, tree_id):
        return self.get_node(tree_id, "start")

    def step(self, tree_id, node_id, user_text):
        """
        Avança na árvore:
          - QUESTION: tenta casar user_text com rótulos; se der, retorna next_node.
          - LEAF: retorna soluções.
        Retorno (dict):
          - kind: "QUESTION"|"LEAF"
          - text: pergunta (QUESTION) ou texto opcional (LEAF)
          - options: [labels] (QUESTION)
          - next_node (QUESTION, quando houve match)
          - solution_q3 / solution_q4 (LEAF)
          - error (se algo errado)
        """
        nodes = self.trees.get(tree_id) or {}
        node = nodes.get(node_id)
        if not node:
            return {"error": f"Nó inexistente: {tree_id}.{node_id}"}

        if node["kind"] == "LEAF":
            return {
                "kind": "LEAF",
                "text": node.get("text", ""),
                "solution_q3": node.get("sol_q3", ""),
                "solution_q4": node.get("sol_q4", "")
            }

        # QUESTION
        labels = [opt["label"] for opt in node["options"] if opt["label"]]
        if not labels:
            return {"error": f"Nó QUESTION sem opções: {tree_id}.{node_id}"}

        # se usuário não digitou nada (ou digitou livre), ofereça opções
        if not user_text.strip():
            return {"kind": "QUESTION", "text": node["text"], "options": labels}

        # fuzzy matching com os rótulos
        best = process.extractOne(user_text, labels, scorer=fuzz.token_set_ratio)
        if not best or best[1] < 50:  # ↓ antes 60
            return {
                "kind": "QUESTION",
                "text": node["text"],
                "options": labels
            }


        chosen_label = best[0]
        next_id = ""
        for opt in node["options"]:
            if opt["label"] == chosen_label:
                next_id = opt["next"]
                break

        if not next_id:
            return {"error": f"Opção sem próximo nó em {tree_id}.{node_id}"}

        return {
            "kind": "QUESTION",
            "text": node["text"],
            "options": labels,
            "next_node": next_id,
            "chosen": chosen_label
        }
