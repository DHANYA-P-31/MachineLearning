import numpy as np
import streamlit as st


st.set_page_config(page_title="K-Bandit", layout="centered")
st.title("K-Armed Bandit")
st.caption("Epsilon-greedy action selection")

k = st.slider("Number of arms (k)", 2, 10, 5)
steps = st.slider("Steps", 50, 2000, 500, 50)
epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.1)

if st.button("Run", type="primary"):
    np.random.seed(42)

    true_means = np.random.normal(0, 1, k)
    q_est = np.zeros(k)
    counts = np.zeros(k)

    rewards = []
    actions = []

    for _ in range(steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(k)
        else:
            a = int(np.argmax(q_est))

        r = np.random.normal(true_means[a], 1)
        counts[a] += 1
        q_est[a] += (r - q_est[a]) / counts[a]

        rewards.append(r)
        actions.append(a)

    avg_reward = float(np.mean(rewards))
    best_arm = int(np.argmax(true_means))
    chosen_best_rate = float(np.mean(np.array(actions) == best_arm))

    st.write("True means:", np.round(true_means, 3))
    st.write("Estimated means:", np.round(q_est, 3))
    st.write("Average reward:", round(avg_reward, 4))
    st.write("Best arm chosen rate:", round(chosen_best_rate, 4))
