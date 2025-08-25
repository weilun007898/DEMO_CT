"""
Order Assistant Demo (FastAPI + OpenAI Tool Calling + Firebase Realtime DB)
----------------------------------------------------------------------------
- POST /chat      : Chat endpoint (OpenAI tool calling wired to Firebase)
- GET  /          : Minimal chat webpage (inline HTML/CSS/JS)
- GET  /health    : Health check
- GET  /static/...: Serves embedded CSS/JS

Env Vars:
  OPENAI_API_KEY=sk-...
  FIREBASE_RTDB_URL=https://<your-db>.firebaseio.com
  # One of:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccountKey.json
  # or
  FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'

Run:
  pip install fastapi uvicorn "openai>=1.30.0" pydantic firebase-admin python-dotenv
  python order_assistant_firebase_app.py
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
##try
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel
import uvicorn

# OpenAI SDK (v1.x)
try:
    from openai import OpenAI
    _openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=_openai_api_key) if _openai_api_key else None
except Exception:
    client = None

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------
# In-memory chat session state
# ---------------------------
SESSIONS: Dict[str, List[Dict[str, Any]]] = {}

SYSTEM_PROMPT = """You are an order assistant for a lens supplier.
You can:
- Register/create customers
- Place orders
- Modify existing orders (add/remove items, change quantities, change shipping address)
- Update a customer's default address
- Provide order status

Use the provided tools (functions) whenever the user request matches a DB action.
If the user asks to "register", "sign up", or "create an account", treat it as customer creation and call create_customer. Ask for missing details like name, email, and address.
Ask for missing details (e.g., customer name/email, shipping address, SKU, quantity).
Summarize critical changes (e.g., address changes) if the user looks uncertain.
Keep replies concise and helpful.

Known SKUs (demo):
- 豹纹短裙 003
- POLO T 007
- Ultraman 内裤 008

Assume currency is SGD. Payments and taxes handled elsewhere.
"""

# ---------------
# Firebase Setup
# ---------------

def init_firebase():
    """Initialize Firebase Admin from environment variables.

    Expected env vars for deployment:
      - FIREBASE_RTDB_URL: your RTDB URL
      - FIREBASE_SERVICE_ACCOUNT_JSON: full JSON of service account (preferred)
        OR
      - GOOGLE_APPLICATION_CREDENTIALS: path to a service account JSON file

    Falls back to the original local dev path if env vars are not provided and the
    file exists (useful for local development only).
    """
    if firebase_admin._apps:
        return

    database_url = os.getenv("FIREBASE_RTDB_URL")
    service_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    cred = None
    if service_json:
        try:
            cred_info = json.loads(service_json)
            cred = credentials.Certificate(cred_info)
        except Exception as e:
            raise RuntimeError(f"Invalid FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
    elif cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
    else:
        # Local fallback (dev convenience)
        default_path = r"C:\Users\shtwa\Desktop\demoforct-firebase-adminsdk-fbsvc-62e64ec5a6.json"
        if os.path.exists(default_path):
            cred = credentials.Certificate(default_path)
            if not database_url:
                database_url = "https://demoforct-default-rtdb.asia-southeast1.firebasedatabase.app"

    if not cred:
        raise RuntimeError(
            "Firebase credentials not provided. Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS."
        )
    if not database_url:
        raise RuntimeError("FIREBASE_RTDB_URL not set.")

    firebase_admin.initialize_app(cred, {"databaseURL": database_url})


# ---------------------------------
# Firebase helper functions (RTDB)
# ---------------------------------
def customers_ref():
    return db.reference("customers")

def orders_ref():
    return db.reference("orders")

def get_customer_by_id(customer_id: str) -> Optional[Dict[str, Any]]:
    data = customers_ref().child(customer_id).get()
    if not data:
        return None
    data["id"] = customer_id
    return data

def find_customer_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Find a customer by email. Falls back to a full scan if index is missing."""
    norm_email = (email or "").strip().lower()
    if not norm_email:
        return None

    snap = None
    try:
        # Fast path when RTDB rules define {"customers": {".indexOn": ["email"]}}
        snap = customers_ref().order_by_child("email").equal_to(norm_email).get()
    except Exception:
        # Missing index or query not supported -> fall back to scan below
        snap = None

    if snap:
        for cid, cust in snap.items():  # {id: customer}
            cust = dict(cust)
            cust["id"] = cid
            return cust

    # Fallback: scan all customers (fine for demos/small datasets)
    all_cust = customers_ref().get() or {}
    for cid, cust in all_cust.items():
        stored_email = (cust.get("email") or "").strip().lower()
        if stored_email == norm_email:
            cust = dict(cust)
            cust["id"] = cid
            return cust
    return None

def find_customer_by_name_address(name: str, address: str) -> Optional[Dict[str, Any]]:
    # Simple scan (ok for demo). For production, consider a secondary index path.
    all_cust = customers_ref().get() or {}
    for cid, cust in all_cust.items():
        if (cust.get("name") == name) and (cust.get("address") == address):
            cust = dict(cust)
            cust["id"] = cid
            return cust
    return None

def find_or_create_customer(name: Optional[str], email: Optional[str], address: Optional[str]) -> Dict[str, Any]:
    # Normalize inputs to reduce duplicate records caused by whitespace/case
    norm_name = (name or "").strip()
    norm_email = (email or "").strip().lower()
    norm_address = (address or "").strip()

    cust = None
    # Prefer lookup by email when available (acts as a unique key in demo)
    if norm_email:
        cust = find_customer_by_email(norm_email)

    # Fallback dedupe by exact name+address match
    if cust is None and norm_name and norm_address:
        cust = find_customer_by_name_address(norm_name, norm_address)

    # Create if not found and we have at least a name or an email
    if cust is None and (norm_name or norm_email):
        cid = str(uuid.uuid4())
        cust_obj = {
            "id": cid,
            "name": norm_name,
            "email": norm_email,
            "address": norm_address
        }
        try:
            print("try to create customer")
            customers_ref().child(cid).set(cust_obj)
        except Exception as e:
            # Surface Firebase errors clearly so the chat can report root cause
            raise RuntimeError(f"Firebase write failed while creating customer: {e}")
        cust = cust_obj

    if cust is None:
        raise ValueError("Cannot resolve customer. Provide at least name or email.")
    return cust

def update_customer_address(customer_id: str, new_address: str) -> Dict[str, Any]:
    ref = customers_ref().child(customer_id)
    if not ref.get():
        raise ValueError("Customer not found")
    ref.update({"address": new_address})
    cust = ref.get()
    cust["id"] = customer_id
    return cust

def create_order(customer_id: str, items: List[Dict[str, Any]], shipping_address: Optional[str]) -> Dict[str, Any]:
    if not items:
        raise ValueError("Items cannot be empty.")
    if shipping_address is None:
        cust = get_customer_by_id(customer_id)
        shipping_address = (cust or {}).get("address", "")

    order_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    order_doc = {
        "id": order_id,
        "customer_id": customer_id,
        "status": "PLACED",
        "items": items,  # list of {sku, quantity}
        "shipping_address": shipping_address or "",
        "created_at": created_at
    }
    orders_ref().child(order_id).set(order_doc)
    return order_doc

def get_order(order_id: str) -> Dict[str, Any]:
    data = orders_ref().child(order_id).get()
    if not data:
        raise ValueError("Order not found")
    return data

def save_order(order: Dict[str, Any]) -> Dict[str, Any]:
    oid = order["id"]
    orders_ref().child(oid).update({
        "customer_id": order["customer_id"],
        "status": order.get("status", "PLACED"),
        "items": order.get("items", []),
        "shipping_address": order.get("shipping_address", "")
    })
    return get_order(oid)

# -----------------------
# Tool (function) schemas
# -----------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_customer",
            "description": "Create or fetch a customer. Use when user wants to register/create account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "address": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Create a new order for a customer with item list and optional shipping address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "object",
                        "description": "Customer identification info. Provide at least email or name.",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "address": {"type": "string"}
                        }
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1}
                            },
                            "required": ["sku", "quantity"]
                        },
                        "minItems": 1
                    },
                    "shipping_address": {
                        "type": "string",
                        "description": "If omitted, uses customer's default address."
                    }
                },
                "required": ["items"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_order",
            "description": "Modify an existing order: add/remove/update items or change shipping address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "add_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1}
                            },
                            "required": ["sku", "quantity"]
                        }
                    },
                    "remove_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SKUs to remove entirely."
                    },
                    "update_quantities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 0}
                            },
                            "required": ["sku", "quantity"]
                        },
                        "description": "If quantity becomes 0, the item is removed."
                    },
                    "new_shipping_address": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_customer_address",
            "description": "Update a customer's default address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "new_address": {"type": "string"}
                },
                "required": ["customer_id", "new_address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Fetch the current status and details of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    }
]

# -----------------------
# Tool function execution
# -----------------------
def tool_create_customer(args: Dict[str, Any]) -> Dict[str, Any]:
    customer = find_or_create_customer(
        name=args.get("name"),
        email=args.get("email"),
        address=args.get("address"),
    )
    return {
        "ok": True,
        "customer": {
            "id": customer["id"],
            "name": customer.get("name", ""),
            "email": customer.get("email", ""),
            "address": customer.get("address", ""),
        },
    }

def tool_place_order(args: Dict[str, Any]) -> Dict[str, Any]:
    customer_input = args.get("customer", {}) or {}
    items = args.get("items", []) or []
    shipping_address = args.get("shipping_address")

    cust_id = customer_input.get("id")
    if cust_id:
        customer = get_customer_by_id(cust_id)
        if not customer:
            raise ValueError("Customer id not found.")
    else:
        customer = find_or_create_customer(
            name=customer_input.get("name"),
            email=customer_input.get("email"),
            address=customer_input.get("address")
        )
    order = create_order(customer["id"], items, shipping_address)
    return {
        "ok": True,
        "order": {
            "id": order["id"],
            "status": order["status"],
            "items": order.get("items", []),
            "shipping_address": order.get("shipping_address", ""),
            "created_at": order.get("created_at", ""),
            "customer_id": order["customer_id"]
        }
    }

def tool_modify_order(args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = args["order_id"]
    add_items = args.get("add_items") or []
    remove_items = set(args.get("remove_items") or [])
    updates = args.get("update_quantities") or []
    new_shipping_address = args.get("new_shipping_address")

    order = get_order(order_id)
    current_items = {i["sku"]: int(i["quantity"]) for i in (order.get("items") or [])}

    # Add items
    for it in add_items:
        sku = it["sku"]
        qty = int(it["quantity"])
        current_items[sku] = current_items.get(sku, 0) + qty

    # Remove items
    for sku in remove_items:
        current_items.pop(sku, None)

    # Update quantities
    for upd in updates:
        sku = upd["sku"]
        qty = int(upd["quantity"])
        if qty <= 0:
            current_items.pop(sku, None)
        else:
            current_items[sku] = qty

    order["items"] = [{"sku": s, "quantity": q} for s, q in current_items.items()]
    if new_shipping_address:
        order["shipping_address"] = new_shipping_address

    saved = save_order(order)
    return {
        "ok": True,
        "order": {
            "id": saved["id"],
            "status": saved.get("status", "PLACED"),
            "items": saved.get("items", []),
            "shipping_address": saved.get("shipping_address", ""),
            "customer_id": saved.get("customer_id", "")
        }
    }

def tool_update_customer_address(args: Dict[str, Any]) -> Dict[str, Any]:
    cust = update_customer_address(args["customer_id"], args["new_address"])
    return {"ok": True, "customer": cust}

def tool_get_order_status(args: Dict[str, Any]) -> Dict[str, Any]:
    order = get_order(args["order_id"])
    return {
        "ok": True,
        "order": {
            "id": order["id"],
            "status": order.get("status", "PLACED"),
            "items": order.get("items", []),
            "shipping_address": order.get("shipping_address", ""),
            "customer_id": order.get("customer_id", ""),
            "created_at": order.get("created_at", "")
        }
    }

TOOL_IMPLS = {
    "create_customer": tool_create_customer,
    "place_order": tool_place_order,
    "modify_order": tool_modify_order,
    "update_customer_address": tool_update_customer_address,
    "get_order_status": tool_get_order_status
}

# -----------------------
# OpenAI chat orchestration
# -----------------------
def ensure_session(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every message has a string content (no nulls) before sending to OpenAI."""
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        if m.get("content") is None:
            m["content"] = ""
        sanitized.append(m)
    return sanitized

def call_openai(messages: List[Dict[str, Any]]):
    # Allow tuning the model and token budget via env vars
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=_sanitize_messages(messages),
        tools=TOOLS,
        tool_choice="auto"
    )
    return resp

def chat_step(session_id: str, user_message: str) -> str:
    ensure_session(session_id)
    messages = SESSIONS[session_id]
    messages.append({"role": "user", "content": user_message})

    response = call_openai(messages)
    msg = response.choices[0].message

    # If tools are requested, execute them and then let the model summarize
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_errors: List[str] = []
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            impl = TOOL_IMPLS.get(fn_name)
            if not impl:
                tool_result = {"ok": False, "error": f"Unknown tool {fn_name}"}
            else:
                try:
                    tool_result = impl(args)
                except Exception as e:
                    tool_result = {"ok": False, "error": str(e)}

            messages.append({
                "role": "assistant",
                "tool_calls": [tc],
                "content": ""
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": json.dumps(tool_result)
            })

            if not tool_result.get("ok", False):
                tool_errors.append(f"{fn_name}: {tool_result.get('error', 'Unknown error')}")

        if tool_errors:
            error_text = "Error while performing the requested action: " + " | ".join(tool_errors)
            messages.append({"role": "assistant", "content": error_text})
            return error_text

        final_resp = call_openai(messages)
        final_msg = final_resp.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})
        return final_msg.content or "(no content)"
    else:
        messages.append({"role": "assistant", "content": msg.content})
        return msg.content or "(no content)"

# -----------------------
# FastAPI App & Routes
# -----------------------
app = FastAPI(title="Order Assistant Demo (Firebase RTDB)")

# CORS: allow your local dev + future static hosts; for demos, allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure Firebase is initialized when the app starts (works with `uvicorn DEMO_CT.view:app`)
@app.on_event("startup")
def _startup_init_services():
    init_firebase()

class ChatIn(BaseModel):
    session_id: str
    message: str

class ChatOut(BaseModel):
    reply: str

@app.get("/health")
def health():
    ok = (client is not None) and bool(firebase_admin._apps)
    return {"ok": ok}

@app.post("/chat", response_model=ChatOut)
def chat(incoming: ChatIn):
    if client is None:
        return ChatOut(reply="OpenAI client not initialized. Set OPENAI_API_KEY.")
    try:
        reply = chat_step(incoming.session_id, incoming.message)
        return ChatOut(reply=reply)
    except Exception as e:
        return JSONResponse({"reply": f"Error: {e}"}, status_code=500)

# -----------------------
# Inline Web UI (static)
# -----------------------
HTML_CONTENT = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Order Assistant Demo</title>
  <link rel="stylesheet" href="/static/styles.css" />
  <link rel="icon" href="data:," />
</head>
<body>
  <header class="app-header">
    <div class="brand">
      <span class="dot"></span>
      <h1>Order Assistant</h1>
    </div>
    <div class="status" id="connStatus">Connected</div>
  </header>

  <main class="container">
    <div class="chat" id="chat"></div>


    <form id="chatForm" class="composer">
      <input id="messageInput" type="text" placeholder="Type your message…" autocomplete="off" required />
      <button id="sendBtn" type="submit">Send</button>
    </form>
  </main>

  <footer class="app-footer">
    <small>Demo only. Do not enter sensitive data.</small>
  </footer>

  <script>
    // If you deploy frontend separately, set your backend URL here:
    // const API_BASE = "https://your-backend.example.com";
    const API_BASE = window.location.origin;
  </script>
  <script src="/static/app.js"></script>
</body>
</html>
"""

CSS_CONTENT = """:root {
  --bg: #0b1020;
  --panel: #121830;
  --panel-2: #0e1430;
  --text: #e8eefc;
  --muted: #a7b0c2;
  --accent: #5aa2ff;
  --accent-2: #4bd1c5;
  --warning: #ffcc66;
  --danger: #ff6b6b;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
  background: radial-gradient(60vw 60vh at 10% 10%, #0b1430, #0b1020 60%);
  color: var(--text);
}
.app-header, .app-footer {
  padding: 12px 16px;
  background: linear-gradient(90deg, var(--panel), var(--panel-2));
  border-bottom: 1px solid #1e2548;
}
.app-footer { border-top: 1px solid #1e2548; border-bottom: none; opacity: .85; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand h1 { font-size: 18px; margin: 0; font-weight: 600; letter-spacing: .3px; }
.brand .dot { width: 10px; height: 10px; border-radius: 50%; background: linear-gradient(45deg, var(--accent), var(--accent-2)); box-shadow: 0 0 16px var(--accent); }
.status { font-size: 12px; color: var(--muted); float: right; margin-top: -18px; }
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
  display: grid;
  grid-template-rows: 1fr auto auto;
  gap: 12px;
  min-height: calc(100vh - 110px);
}
.chat {
  background: rgba(10, 14, 30, 0.55);
  backdrop-filter: blur(6px);
  border: 1px solid #1b2347;
  border-radius: 12px;
  padding: 16px;
  overflow-y: auto;
  min-height: 50vh;
}
.msg {
  display: flex;
  margin: 8px 0;
  gap: 10px;
  align-items: flex-start;
}
.msg .avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  flex: 0 0 28px;
  background: linear-gradient(45deg, var(--accent), var(--accent-2));
  opacity: .85;
}
.msg.user .avatar { background: linear-gradient(45deg, #8b9ccf, #93a3ff); }
.msg .bubble {
  padding: 10px 12px;
  border-radius: 12px;
  max-width: 80%;
  line-height: 1.3;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.msg.user .bubble { background: #303b70; border: 1px solid #3b4a8a; }
.msg.assistant .bubble { background: #1a2247; border: 1px solid #243062; }
.msg.system .bubble {
  background: #432b11;
  border: 1px solid #5d3d17;
  color: #ffd9a3;
}
.suggestions { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  background: #17204a;
  color: var(--text);
  border: 1px solid #223064;
  border-radius: 999px;
  padding: 8px 12px;
  cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.composer { display: grid; grid-template-columns: 1fr auto; gap: 10px; }
.composer input {
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid #1d2652;
  background: #0f1736;
  color: var(--text);
  outline: none;
}
.composer input:focus { border-color: var(--accent); }
.composer button {
  padding: 12px 16px;
  border-radius: 10px;
  border: 1px solid #1d2652;
  background: linear-gradient(45deg, var(--accent), var(--accent-2));
  color: #0b1020;
  font-weight: 600;
  cursor: pointer;
}
.loading {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,.2);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
small { color: var(--muted); }
"""

JS_CONTENT = """(function () {
  const API_BASE = window.API_BASE || window.location.origin;
  const CHAT_URL = `${API_BASE}/chat`;
  const SESSION_KEY = "orderbot_session_id";
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now());
    localStorage.setItem(SESSION_KEY, sessionId);
  }

  const chat = document.getElementById("chat");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const connStatus = document.getElementById("connStatus");
  const suggestions = document.querySelectorAll(".chip");

  function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text) e.textContent = text;
    return e;
  }

  function addMessage(role, content) {
    const row = el("div", `msg ${role}`);
    const avatar = el("div", "avatar");
    const bubble = el("div", "bubble");
    bubble.textContent = content;
    row.appendChild(avatar);
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
  }

  function setBusy(busy) {
    input.disabled = busy;
    sendBtn.disabled = busy;
    sendBtn.textContent = busy ? "Sending…" : "Send";
  }

  async function sendMessage(text) {
    if (!text.trim()) return;
    addMessage("user", text);
    setBusy(true);

    const thinkingRow = el("div", "msg assistant");
    const avatar = el("div", "avatar");
    const bubble = el("div", "bubble");
    const spinner = el("span", "loading");
    bubble.appendChild(spinner);
    bubble.append(" Thinking…");
    thinkingRow.appendChild(avatar);
    thinkingRow.appendChild(bubble);
    chat.appendChild(thinkingRow);
    chat.scrollTop = chat.scrollHeight;

    try {
      const res = await fetch(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      let data = null;
      try { data = await res.json(); } catch {}
      if (!res.ok) {
        const serverMsg = data && data.reply ? data.reply : `HTTP ${res.status}`;
        throw new Error(serverMsg);
      }
      thinkingRow.remove();
      addMessage("assistant", (data && data.reply) ? data.reply : "(no reply)");
      connStatus.textContent = "Connected";
      connStatus.style.color = "#a7b0c2";
    } catch (err) {
      thinkingRow.remove();
      addMessage("system", `Error: ${err.message || err}`);
      connStatus.textContent = "Connection error";
      connStatus.style.color = "#ffcc66";
    } finally {
      setBusy(false);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value;
    input.value = "";
    sendMessage(text);
  });

  suggestions.forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.textContent;
      input.focus();
    });
  });

  addMessage("assistant", "Hi! I can place and modify orders, and update addresses. How can I help?");
})();"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

@app.get("/static/styles.css")
def static_css():
    return Response(CSS_CONTENT, media_type="text/css")

@app.get("/static/app.js")
def static_js():
    return Response(JS_CONTENT, media_type="application/javascript")

# ---------------
# Bootstrap + Run
# ---------------
if __name__ == "__main__":
    init_firebase()
    print("✅ Firebase RTDB ready. OpenAI client:", "OK" if client else "NOT INITIALIZED")
    print("🚀 Starting server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
