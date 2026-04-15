import streamlit as st
import pandas as pd

def is_consistent(hypothesis, instance):
    for h,i in zip(hypothesis,instance):
        if h!="?" and h !=i:
            return False
    return True

def more_general(h1,h2):
    for a,b in zip(h1,h2):
        if a!="?" and (a!=b):
            return False
    return True

def candidate_elimination(df,label_col="Approval"):
    num_attributes = len(df.columns) - 1
    S = ["0"] * num_attributes
    G = [["?"] * num_attributes]
    for index,rows in df.iterrows():
        instance = rows.iloc[:-1].tolist()
        label = rows.iloc[-1]
        if label == "Yes":
            for i in range(num_attributes):
                if S[i] == "0":
                    S[i]=  instance[i]
                elif instance[i] != S[i]:
                    S[i] = "?"
            G = [g for g in G if is_consistent(g,instance)]
        else:
            G_new = []
            for g in G:
                if is_consistent(g,instance):
                    for i in range(num_attributes):
                        if g[i] == "?" and S[i] != "?" and S[i] != instance[i]:
                            new_g = g.copy()
                            new_g[i] = S[i]
                            G_new.append(new_g)
                else:
                    G_new.append(g)
            G = G_new
        G = [g for g in G if not any(more_general(g2,g) and g2 != g for g2 in G)]
        st.write(f"After instance {index+1} ({label}):")
        st.write("S:", S)
        st.write("G:", G)
    return df.columns[:-1].tolist(), S, G

def predict(instance,s,g):
    s_match = is_consistent(s,instance)
    g_match = all(is_consistent(hyp,instance) for hyp in g)
    if s_match and g_match:
        return "Yes"
    elif not s_match and not g_match:
        return "No"
    else:
        return "Unknown"

st.set_page_config(page_title="Candidate Elimination Algorithm", page_icon = "ML")
st.title("Candidate Elimination Algorithm")
st.caption("This app implements the Candidate Elimination Algorithm for concept learning.")
st.subheader("Upload Dataset")
uploaded = st.file_uploader("Upload a CSV file with the last column as the label.",type = ["csv","xlsx"])
if uploaded is not None:
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
    st.write("Dataset:")
    st.dataframe(df)
    if st.button("Train",type = "primary"):
        st.subheader("Running Candidate Elimination")
        attributes, S, G = candidate_elimination(df)
        st.session_state["attrs"] = attributes
        st.session_state["S"] = S
        st.session_state["G"] = G
    if all(k in st.session_state for k in ["attrs", "S", "G"]):
        st.subheader("Model Boundaries")
        st.write("S:", st.session_state["S"])
        st.write("G:", st.session_state["G"])
        st.subheader("Predict New Instance")
        values = []
        for attr in st.session_state["attrs"]:
                values.append(st.text_input(attr, key=f"input_{attr}"))
        if st.button("Predict"):
            result = predict(values, st.session_state["S"], st.session_state["G"])
            st.success(f"Prediction: {result}")            