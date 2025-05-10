import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from prophet import Prophet
from datetime import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

# Configuração geral da páginay
st.set_page_config(
    page_title='Análise Preço do Petróleo Brent',
    page_icon=":chart_with_upwards_trend:",
    layout='wide'
)

# CSS personalizado para remover scroll da sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            overflow-y: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# === BARRA LATERAL ===
with st.sidebar:
    # Logo
    st.image(
        "https://impactospositivos.com/wp-content/uploads/2024/03/FIAP-Apoiador.png",
        caption="Pós-Tech FIAP | Tech Challenge Fase 4 | Grupo 5",
        width=220
    )

    # Menu de navegação
    escolha = option_menu(
        "Tech Challenge: Fase 4",
        ["Exploração e Insights", "Deploy", "Conclusão", "Referências"],
        icons=["bar-chart-line", "gear", "check2-square", "book"],
        menu_icon="laptop",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0px", "background-color": "#0e1117"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#6c63ff"},
        }
    )

    # Informações do grupo
    st.title('Grupo 5 - FIAP')
    st.write('''Integrantes:
- Anderson Silva
- Kelvyn
- Evandro Godin
- Sandra
- Michael''')

# === CABEÇALHO PRINCIPAL ===
st.header('Análise Preço do Petróleo *Brent*')
st.markdown('**Pós-Tech FIAP Data Analytics** | Tech Challenge Fase 4 | Grupo 5')

# === CONTEÚDO PRINCIPAL ===
if escolha == "Exploração e Insights":
    st.header("📊 Exploração e Insights")
    st.write("Nesta seção, apresentamos a análise exploratória dos dados históricos do petróleo Brent.")

    aba1, aba2, aba3 = st.tabs(['Introdução', 'Objetivos', 'Insights'])

    with aba1:
        st.subheader('Contextualização histórica')
        st.write(
            "O petróleo é uma das commodities mais influentes no mercado global, com grande impacto "
            "econômico e geopolítico desde sua descoberta no século XIX. O tipo *Brent* é usado como "
            "referência para os preços no mercado internacional."
        )

    with aba2:
        st.subheader('Objetivos do estudo')
        st.markdown("""
        - Analisar a flutuação do preço do petróleo entre 1987 e 2024  
        - Criar um dashboard interativo para gerar 4 insights sobre a variação de preço  
        - Desenvolver um modelo de previsão de preço para 90 dias  
        - Prever o preço do petróleo com input do usuário utilizando Machine Learning
        """)

    with aba3:
        st.subheader("Insights sobre o Petróleo")
        components.iframe(
            "https://app.powerbi.com/view?r=eyJrIjoiMDhlMDM3ZTItODlhNy00MGU5LWJlYWEtNWVlNzE0NDk5MTBiIiwidCI6IjcwNmUwZTIyLTUwZjktNDI2Ni1iOGMxLWViNDIyNmNkZDllYSJ9",
            width=2000,
            height=600,
            scrolling=True
        )

elif escolha == "Deploy":
    st.header("⚙️ Deploy")
    st.write("Nesta seção, apresentamos como o modelo foi preparado para produção e as ferramentas utilizadas no deploy.")

elif escolha == "Conclusão":
    st.header("✅ Conclusão")
    st.write("Resumo dos principais aprendizados, desafios enfrentados e sugestões para trabalhos futuros.")

elif escolha == "Referências":
    st.header("📚 Referências")
    st.write("Fontes de dados, artigos, livros e materiais utilizados no projeto.")
