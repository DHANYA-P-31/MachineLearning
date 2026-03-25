import numpy as np
import streamlit as st


st.set_page_config(page_title="Q-Learning", layout="centered")
st.title("Q-Learning (Simple Grid)")
st.caption("Small 1D environment with Q-table update")

n_states = 5
n_actions = 2  # 0=left, 1=right
start_state = 0
goal_state = n_states - 1

alpha = st.slider("Learning rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.slider("Discount factor (gamma)", 0.01, 0.99, 0.9)
epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.2)
episodes = st.slider("Episodes", 10, 500, 100, 10)

if st.button("Train", type="primary"):
    Q = np.zeros((n_states, n_actions), dtype=float)

    for _ in range(episodes):
        state = start_state
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(Q[state]))

            if action == 0:
                next_state = max(0, state - 1)
            else:
                next_state = min(n_states - 1, state + 1)

            reward = 1.0 if next_state == goal_state else -0.01
            done = next_state == goal_state

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state

    policy = ["Left" if int(np.argmax(Q[s])) == 0 else "Right" for s in range(n_states)]

    st.write("Q-table:")
    st.dataframe(Q)
    st.write("Learned policy:", policy)
