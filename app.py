import streamlit as st
import joblib
import numpy as np

modelo = joblib.load("modelo.pkl")

st.set_page_config(page_title="Sobrevivência no Titanic", page_icon="🚢")
st.title("🚢 Titanic — Previsão de Sobrevivência")
st.markdown("Preencha os dados do passageiro para descobrir se ele sobreviveria.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Classe do navio (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}ª Classe")
    sexo = st.radio("Sexo", ["Masculino", "Feminino"])
    idade = st.slider("Idade", min_value=1, max_value=100, value=30)

with col2:
    sibsp = st.number_input("Irmãos / Cônjuges a bordo (SibSp)", min_value=0, max_value=10, value=0)

sex_num = 1 if sexo == "Feminino" else 0

st.divider()

if st.button("🔍 Prever Sobrevivência", use_container_width=True):
    entrada = np.array([[pclass, sex_num, idade, sibsp]])
    predicao = modelo.predict(entrada)[0]
    probabilidade = modelo.predict_proba(entrada)[0]

    if predicao == 1:
        st.success("✅ O passageiro **SOBREVIVERIA**!")
        st.metric("Probabilidade de sobrevivência", f"{probabilidade[1]:.1%}")
    else:
        st.error("❌ O passageiro **NÃO sobreviveria**.")
        st.metric("Probabilidade de não sobrevivência", f"{probabilidade[0]:.1%}")

    st.progress(float(probabilidade[1]), text="Chance de sobrevivência")
