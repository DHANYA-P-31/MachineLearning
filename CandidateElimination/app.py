import pandas as pd
import streamlit as st

def is_consistent(hypothesis, instance):
    for h, i in zip(hypothesis, instance):
        if h != "?" and h != i:
            return False
    return True

def more_general(h1, h2):
    for a, b in zip(h1, h2):
        if a != "?" and (a != b):
            return False
    return True


def candidate_elimination(df, label_col="Approval"):
	num_attributes = len(df.columns) - 1
	S = ["0"] * num_attributes
	G = [["?"] * num_attributes]

	for index, row in df.iterrows():
		instance = row.iloc[:-1].tolist()
		label = row.iloc[-1]

		if label == "Yes":
			for i in range(num_attributes):
				if S[i] == "0":
					S[i] = instance[i]
				elif instance[i] != S[i]:
					S[i] = "?"
			G = [g for g in G if is_consistent(g, instance)]
		else:
			G_new = []
			for g in G:
				if is_consistent(g, instance):
					for i in range(num_attributes):
						if g[i] == "?" and S[i] != "?" and S[i] != instance[i]:
							new_g = g.copy()
							new_g[i] = S[i]
							G_new.append(new_g)
				else:
					G_new.append(g)
			G = G_new

		G = [
			g for g in G
			if not any(more_general(g2, g) and g2 != g for g2 in G)
		]
	return df.columns[:-1].tolist(), S, G


def predict(instance, s, g):
	s_match = is_consistent(s, instance)
	g_match = all(is_consistent(hyp, instance) for hyp in g)
	if s_match and g_match:
		return "Yes"
	if not g_match:
		return "No"
	return "Uncertain"


st.set_page_config(page_title="Candidate Elimination", page_icon="ML")
st.title("Candidate Elimination - Minimal Predictor")
st.caption("Upload data, train boundaries S/G, then predict a new instance.")

uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded is not None:
	if uploaded.name.lower().endswith(".csv"):
		data = pd.read_csv(uploaded)
	else:
		data = pd.read_excel(uploaded)

	st.subheader("Data Preview")
	st.dataframe(data, use_container_width=True)

	default_label = "Approval" if "Approval" in data.columns else data.columns[-1]
	label_col = st.selectbox("Label column", options=data.columns.tolist(), index=data.columns.tolist().index(default_label))

	if st.button("Train", type="primary"):
		attrs, s_final, g_final = candidate_elimination(data, label_col=label_col)
		st.session_state["attrs"] = attrs
		st.session_state["S"] = s_final
		st.session_state["G"] = g_final

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
else:
	st.info("Upload a file to start.")
