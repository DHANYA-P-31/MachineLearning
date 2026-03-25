from pathlib import Path

import joblib
import numpy as np
import streamlit as st


st.set_page_config(page_title="HMM Forward + Viterbi", layout="centered")
st.title("Simple HMM UI")
st.caption("Loads hmm_model.joblib saved from hmm.ipynb")

artifact = joblib.load(Path("hmm_model.joblib"))
model = artifact["model"]
obs_to_num = artifact["obs_to_num"]
state_names = artifact["state_names"]

st.write("Allowed observations:", ", ".join(obs_to_num.keys()))

obs_text = st.text_input("Observation sequence (comma-separated)", "normal,cold,dizzy")

if st.button("Run", type="primary"):
    seq = [x.strip().lower() for x in obs_text.split(",") if x.strip()]

    if not seq:
        st.warning("Enter at least one observation.")
    else:
        unknown = [x for x in seq if x not in obs_to_num]
        if unknown:
            st.error(f"Invalid observations: {unknown}")
        else:
            X = np.array([obs_to_num[s] for s in seq], dtype=float).reshape(-1, 1)

            log_prob = model.score(X)
            forward_prob = float(np.exp(log_prob))

            v_log_prob, v_states = model.decode(X, algorithm="viterbi")
            v_path = [state_names[s] for s in v_states]

            st.write("Observation sequence:", seq)
            st.write("Forward probability:", forward_prob)
            st.write("Viterbi path:", v_path)
            st.write("Viterbi log probability:", float(v_log_prob))
