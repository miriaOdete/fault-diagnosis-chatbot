from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, uuid, traceback, threading, time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from failure_flow import FailureFlowEngine

EXCEL_PATH = "tabela_falhas_unificada.xlsx"
EXCEL_SHEET = "Sheet1"
HOST, PORT, DEBUG = "0.0.0.0", 5000, True

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)
CORS(app)

flow = None
init_error = None
retrain_lock = threading.Lock()


def safe_retrain(max_retries=6, delay=1.5):
    global flow
    for i in range(max_retries):
        try:
            with retrain_lock:
                flow.retrain()
            print("[RETRAIN] Modelo atualizado com sucesso.")
            return True
        except PermissionError:
            print(f"[RETRAIN] Arquivo bloqueado (tentativa {i+1}/{max_retries}). Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print("[RETRAIN] Falha:", type(e).__name__, e)
            return False
    return False


class ExcelChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        try:
            if event.is_directory:
                return
            changed = os.path.abspath(event.src_path)
            target = os.path.abspath(EXCEL_PATH)
            if os.path.basename(changed) == os.path.basename(target):
                print(f"[WATCH] Alteração detectada em: {changed}")
                safe_retrain()
        except Exception as e:
            print("[WATCH] Erro no watcher:", type(e).__name__, e)


def start_watcher():
    observer = Observer()
    watch_dir = os.path.dirname(os.path.abspath(EXCEL_PATH)) or "."
    observer.schedule(ExcelChangeHandler(), watch_dir, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[WATCH] Observando: {watch_dir}\\{os.path.basename(EXCEL_PATH)}")
    return observer


try:
    flow = FailureFlowEngine(excel_path=EXCEL_PATH, sheet=EXCEL_SHEET)
    init_error = None
    start_watcher()
except Exception as e:
    flow = None
    init_error = f"{type(e).__name__}: {e}"
    print("[ERRO] Inicialização:", init_error)

SESSIONS = {}


def new_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "stage": "ASK_MODE",
        "mode": None,
        "effect": None,
        "diag": None,
        "last_options": [],
        "history": [],
    }
    return sid


def reset_session(sid):
    SESSIONS[sid] = {
        "stage": "ASK_MODE",
        "mode": None,
        "effect": None,
        "diag": None,
        "last_options": [],
        "history": [],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    ok = (init_error is None) and (flow is not None)
    cols = {
        "modo": getattr(flow, "col_mode", None) if flow else None,
        "efeito": getattr(flow, "col_effect", None) if flow else None,
        "causa": getattr(flow, "col_cause", None) if flow else None,
        "acao": getattr(flow, "col_action", None) if flow else None,
    }
    return jsonify({"ok": ok, "error": init_error, "columns": cols})


@app.post("/retrain")
def retrain():
    try:
        ok = safe_retrain()
        return jsonify({"ok": ok, "msg": "Modelo atualizado com sucesso." if ok else "Falha ao atualizar."})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"{type(e).__name__}: {e}"}), 500


@app.post("/perguntar")
def perguntar():
    try:
        if init_error:
            return jsonify({"resposta": f"Erro na inicialização: {init_error}"}), 500
        if flow is None:
            return jsonify({"resposta": "Engine não carregada."}), 500

        data = request.get_json(silent=True) or {}
        text = (data.get("pergunta") or "").strip()
        sid = data.get("session_id")
        is_option = bool(data.get("is_option", False))

        if not sid or sid not in SESSIONS:
            sid = new_session()
        sess = SESSIONS[sid]

        if not text:
            return jsonify({"resposta": "Descreva o problema para começarmos.", "session_id": sid})

        if not is_option and sess["stage"] != "ASK_MODE":
            reset_session(sid)
            sess = SESSIONS[sid]

        if sess["stage"] == "ASK_MODE":
            chosen = flow.choose_from(text, flow.modes_unique)
            if chosen:
                sess["mode"] = chosen
                sess["stage"] = "ASK_EFFECT"
                effects = flow.list_effects(chosen)
                msg = f"Identifiquei o modo de falha: {chosen}\nSelecione o efeito de falha:"
                sess["last_options"] = effects[:12]
                sess["history"] += [("user", text), ("bot", msg)]
                return jsonify(
                    {"resposta": msg, "session_id": sid, "options": sess["last_options"]}
                )

            mode, score, suggestions = flow.pick_mode(text)
            if mode:
                sess["mode"] = mode
                sess["stage"] = "ASK_EFFECT"
                effects = flow.list_effects(mode)
                msg = f"Identifiquei o modo de falha: {mode}\nSelecione o efeito de falha:"
                sess["last_options"] = effects[:12]
                sess["history"] += [("user", text), ("bot", msg)]
                return jsonify(
                    {"resposta": msg, "session_id": sid, "options": sess["last_options"]}
                )
            else:
                msg = "Não identifiquei uma correspondência exata de modo. Você quis dizer:"
                opts = (suggestions or [])[:5]
                sess["last_options"] = opts
                sess["history"] += [("user", text), ("bot", msg)]
                return jsonify({"resposta": msg, "session_id": sid, "options": opts})

        if sess["stage"] == "ASK_EFFECT":
            mode = sess["mode"]
            effect = flow.choose_from(text, flow.list_effects(mode)) or text
            sess["effect"] = effect

            diag = flow.build_questions(mode, effect, max_q=5)
            sess["diag"] = diag

            if not diag["questions"]:
                result = flow.finalize(diag, top_k=3)
                sess["stage"] = "DIAG_DONE"
                sess["last_options"] = []
                sess["history"] += [("user", text), ("bot", result)]
                return jsonify({"resposta": result, "session_id": sid})

            q0 = diag["questions"][0]["text"]
            sess["stage"] = "DIAG_ASK"
            sess["last_options"] = ["Sim", "Não"]
            sess["history"] += [("user", text), ("bot", q0)]
            return jsonify({"resposta": q0, "session_id": sid, "options": sess["last_options"]})

        if sess["stage"] == "DIAG_ASK":
            diag = sess["diag"]
            diag = flow.answer_question(diag, text)
            sess["diag"] = diag

            if diag["q_index"] < len(diag["questions"]):
                nxt = diag["questions"][diag["q_index"]]["text"]
                sess["last_options"] = ["Sim", "Não"]
                sess["history"] += [("user", text), ("bot", nxt)]
                return jsonify({"resposta": nxt, "session_id": sid, "options": sess["last_options"]})
            else:
                result = flow.finalize(diag, top_k=3)
                sess["stage"] = "DIAG_DONE"
                sess["last_options"] = []
                sess["history"] += [("user", text), ("bot", result)]
                return jsonify({"resposta": result, "session_id": sid})

        if sess["stage"] == "DIAG_DONE":
            reset_session(sid)
            sess = SESSIONS[sid]

            chosen = flow.choose_from(text, flow.modes_unique)
            if chosen:
                sess["mode"] = chosen
                sess["stage"] = "ASK_EFFECT"
                effects = flow.list_effects(chosen)
                msg = f"Identifiquei o modo de falha: {chosen}\nSelecione o efeito de falha:"
                sess["last_options"] = effects[:12]
                sess["history"] += [("user", text), ("bot", msg)]
                return jsonify(
                    {"resposta": msg, "session_id": sid, "options": sess["last_options"]}
                )

            mode, score, suggestions = flow.pick_mode(text)
            if mode:
                sess["mode"] = mode
                sess["stage"] = "ASK_EFFECT"
                effects = flow.list_effects(mode)
                msg = f"Identifiquei o modo de falha: {mode}\nSelecione o efeito de falha:"
                sess["last_options"] = effects[:12]
                sess["history"] += [("user", text), ("bot", msg)]
                return jsonify(
                    {"resposta": msg, "session_id": sid, "options": sess["last_options"]}
                )
            else:
                msg = "Não identifiquei uma correspondência exata de modo. Você quis dizer:"
                opts = (suggestions or [])[:5]
                sess["last_options"] = opts
                sess["history"] += [("user", text), ("bot", msg)]
                return jsonify({"resposta": msg, "session_id": sid, "options": opts})

        reset_session(sid)
        return jsonify({"resposta": "Descreva o problema para começarmos.", "session_id": sid})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"resposta": f"Erro no servidor: {type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    print("[BOOT] Iniciando Flask...")
    print(f"[INFO] Excel: {os.path.abspath(EXCEL_PATH)} | Sheet: {EXCEL_SHEET}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
