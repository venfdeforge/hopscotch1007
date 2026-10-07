"""Hopscotch Support - Session 1: raw OpenAI SDK + glue code (no LangChain)."""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

st.set_page_config(page_title="Hopscotch Support (Session 1)", page_icon="🛍️")
st.title("🛍️ Hopscotch Support")
st.caption("Session 1 · raw OpenAI SDK + glue code")

# --- 1. API key ------------------------------------------------------------
load_dotenv()  # copies values from the .env file into environment variables
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY not found. Copy `.env.example` to `.env`, "
             "paste your key from https://openrouter.ai/keys, then restart the app.")
    st.stop()  # stop the script here; nothing below runs

# --- 2. Client + model -----------------------------------------------------
# OpenRouter speaks the OpenAI wire format, so we reuse the OpenAI SDK and just
# change base_url.
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
# HARDCODED, and tied to the OpenAI SDK's wire format. Swapping providers is a
# rewrite -> that's the Session 2 problem.
MODEL = "openai/gpt-4o-mini"

# --- 3. System prompt: an f-string gluing role + policy + rules ------------
POLICY = (Path(__file__).parent / "hopscotch_policy.md").read_text(encoding="utf-8")
SYSTEM_PROMPT = f"""You are Hopscotch's customer support assistant for a premium kids' fashion
retailer in Mumbai. Be warm and concise.

=== RETURN POLICY ===
{POLICY}
=== END POLICY ===

Rules:
- Answer ONLY from the policy above. Never invent rules.
- If you need the delivery date, the item's condition, or the order value, ask for it.
- For ANY Section 4 case (child safety, skin reaction/allergy, order above INR 5,000,
  unclear facts, or uncertainty between defect and misuse) do NOT decide. Tell the
  customer their case will be passed to a human agent.
- For defect claims, ask the customer for photo evidence.
"""

# --- 4. Memory --------------------------------------------------------------
# THIS LIST IS THE MEMORY. The WHOLE list is resent on every turn; the model
# itself is stateless and remembers nothing between calls.
# Streamlit re-runs this script top-to-bottom on every message, so a normal
# variable would reset each time. st.session_state survives reruns.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# The classic terminal version is a loop:
#
#   messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#   while True:
#       text = input("You: ")
#       if text.lower() in ("exit", "end", "stop", "bye"):
#           break
#       messages.append({"role": "user", "content": text})
#       reply = client.chat.completions.create(model=MODEL, messages=messages)
#       messages.append({"role": "assistant", "content": reply.choices[0].message.content})
#       print("Bot:", reply.choices[0].message.content)
#
# In Streamlit there is no while-loop: each rerun of this script = ONE loop
# iteration, and st.session_state keeps the list alive between iterations.

# --- 5. Sidebar -------------------------------------------------------------
with st.sidebar:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()
    with st.expander("🔍 Raw history list (what we resend every turn)"):
        st.write(f"Messages in list: {len(st.session_state.messages)}")
        st.json(st.session_state.messages)
    st.markdown("**Example prompts**")
    st.code("I bought a dress 10 days ago, never worn, tags on — can I return it?", language=None)
    st.code("The sole of my son's sneakers peeled off after 3 weeks of school", language=None)
    st.code("A button came off and my toddler nearly put it in his mouth", language=None)
    st.code("My order was ₹7,200, I want a refund", language=None)

# --- 6. Render history (skip the system message) ----------------------------
for m in st.session_state.messages:
    if m["role"] != "system":
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# --- 7. One "loop iteration": new user message -> model -> reply ------------
if user_text := st.chat_input("Ask about returns, refunds, defects…"):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,  # the WHOLE history
                temperature=0.3,
            )
            reply = response.choices[0].message.content  # SDK-specific parsing
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.markdown(reply)
        except Exception as e:
            st.error(f"Something went wrong calling the model: {e}")
            st.session_state.messages.pop()  # drop the unanswered user turn
