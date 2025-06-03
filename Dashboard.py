import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title='Análise Preço do Petróleo Brent',
    page_icon=":chart_with_upwards_trend:",
    layout='wide'
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            overflow-y: hidden;
        }
        .stProgress > div > div > div > div {
            background-color: #6c63ff;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image(
        caption="Pós-Tech FIAP | Tech Challenge Fase 4 | Grupo 5",
        width=220
    )

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

    st.title('Grupo 5 - FIAP')
    st.write('''Integrantes:
- Anderson Silva
- Kelvyn Candido
- Evandro Godin
- Sandra Hoja
- Michael''')

st.header('Análise Preço do Petróleo *Brent*')
st.markdown('**Pós-Tech FIAP Data Analytics** | Tech Challenge Fase 4 | Grupo 5')

@st.cache_data(show_spinner=True)
def fetch_ipeadata():
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Erro ao acessar a página do IpeaData: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'dxgvTable'})
    if not table:
        st.error("Tabela não encontrada.")
        return pd.DataFrame()
    desired_headers = ["Data", "Preço - petróleo bruto - Brent (FOB)"]
    headers = []
    header_row = table.find('tr')
    for th in header_row.find_all('td'):
        header_text = th.text.strip()
        if header_text in desired_headers:
            headers.append(header_text)
    ordered_headers = [h for h in desired_headers if h in headers]
    data = []
    for tr in table.find_all('tr')[1:]:
        row_data = []
        tds = tr.find_all('td', class_='dxgv')
        for td in tds:
            row_data.append(td.text.strip())
        if row_data:
            data.append(row_data)
    df = pd.DataFrame(data, columns=ordered_headers)
    df.columns = ['Data', 'Preço']
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df['Preço'] = pd.to_numeric(df['Preço'].str.replace(",", "."), errors='coerce')
    df.dropna(inplace=True)
    df = df.sort_values('Data').reset_index(drop=True)
    return df

def plot_historico(df):
    fig = px.line(df, x='Data', y='Preço', title='Preço do Petróleo Brent')
    st.plotly_chart(fig, use_container_width=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def lstm_forecast(df):
    window_size = 2
    epochs = 430
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Preço']])
    X, y = [], []
    for i in range(len(data)-window_size-1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+2].flatten())
    X = np.array(X)
    y = np.array(y)
    X_train = torch.FloatTensor(X).view(-1, window_size, 1)
    y_train = torch.FloatTensor(y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    progress_bar = st.progress(0)
    status_text = st.empty()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch+1)/epochs)
        status_text.text(f"Época {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    model.eval()
    last_window = torch.FloatTensor(data[-window_size:]).view(1, window_size, 1).to(device)
    with torch.no_grad():
        pred = model(last_window)
        future_predictions = pred.cpu().numpy().flatten()
    future_predictions = scaler.inverse_transform(future_predictions.reshape(-1,1)).flatten()
    future_dates = pd.date_range(df['Data'].iloc[-1] + timedelta(days=1), periods=2)
    return pd.DataFrame({'Data': future_dates, 'Previsão': future_predictions})

def prophet_forecast(df, steps=90):
    df_prophet = df.rename(columns={'Data': 'ds', 'Preço': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast

if escolha == "Exploração e Insights":
    st.header("📊 Exploração e Insights")
    aba1, aba2, aba3 = st.tabs(['Contexto Mercado', 'Objetivos Técnicos', 'Insights'])

    with aba1:
        st.subheader('Contexto do Mercado de Petróleo Brent')
        st.write("""
        O **Petróleo Brent** é a principal referência global para precificação do petróleo bruto, representando mais de 60% das transações internacionais. 
        Extraído do Mar do Norte, suas características principais incluem:
        
        - **Tipo**: Cru leve e doce (baixo teor de enxofre: 0.37%)
        - **Densidade API**: ~38° (ideal para refino de gasolina e diesel)
        - **Área de Produção**: Campos petrolíferos Brent, Forties, Oseberg e Ekofisk
        
        **Fatores Chave de Influência**:
        1. Decisões da OPEP+ sobre produção
        2. Conflitos geopolíticos em regiões produtoras
        3. Transição energética global e adoção de veículos elétricos
        4. Flutuações cambiais (especialmente USD)
        5. Estoque estratégico de países consumidores
        
        Em 2024, o Brent apresentou volatilidade recorde, variando entre US$ 68 e US$ 92 o barril, segundo dados da EIA. Projeções recentes do DoE americano 
        indicam média de US$ 68 para 2025, com risco de queda para US$ 61 em 2026 devido à expansão de fontes renováveis.
        """)

    with aba2:
        st.subheader('Objetivos Técnicos da Análise')
        st.markdown("""
        **1. Modelagem Preditiva Avançada**  
        - Implementar arquitetura LSTM com 3 camadas (50 unidades) para capturar padrões complexos  
        - Configurar janela temporal ótima de 2 dias baseado em [Wang et al. (2024)]  
        - Treinar com 430 épocas e early stopping para evitar overfitting  

        **2. Análise Comparativa**  
        - Validar resultados contra modelo Prophet com decomposição de sazonalidade  
        - Medir desempenho com métricas:  
          - RMSE (Root Mean Squared Error)  
          - MAE (Mean Absolute Error)  
          - MAPE (Mean Absolute Percentage Error)  

        **3. Aplicação Prática**  
        - Gerar projeções para cenários de curto (30 dias) e médio prazo (90 dias)  
        - Identificar pontos de inflexão críticos na série histórica  
        - Avaliar impacto de eventos extremos (ex: sanções a produtores)  
        """)

    with aba3:
        st.subheader("Visualização Interativa")
        components.iframe(
            "https://app.powerbi.com/view?r=eyJrIjoiMDhlMDM3ZTItODlhNy00MGU5LWJlYWEtNWVlNzE0NDk5MTBiIiwidCI6IjcwNmUwZTIyLTUwZjktNDI2Ni1iOGMxLWViNDIyNmNkZDllYSJ9",
            width=2000,
            height=600
        )


elif escolha == "Previsão e Dados":
    st.header("⚙️ Previsão e Dados")
    df = fetch_ipeadata()
    if not df.empty:
        tab_dados, tab_previsao = st.tabs(["📊 Dados Históricos", "🔮 Previsão"])
        with tab_dados:
            st.subheader("Dados Históricos")
            st.dataframe(df.tail(10))
            plot_historico(df)
        with tab_previsao:
            st.subheader("Escolha o Modelo de Previsão")
            modelo = st.radio("", ('LSTM', 'Prophet'), horizontal=True)
            last_date = df['Data'].iloc[-1].date()
            if modelo == 'LSTM':
                st.write(f"""
                **⚙️ Arquitetura LSTM Otimizada**  
                - **Entrada:** Últimos 10 dias (`window_size = 10`)  
                - **Camadas LSTM:** 1 camada com 50 neurônios  
                - **Épocas:** 430 (com early stopping)  
                - **Data base:** {last_date.strftime('%d/%m/%Y')}  
                - **Previsão:** Próximos 2 dias após a última data da base
                - **Limitação:** Previsões para datas muito distantes podem ser imprecisas, pois o modelo utiliza apenas os valores anteriores para estimar os próximos.
                """)
                
                if st.button("Gerar Previsão com LSTM"):
                    with st.spinner('Treinando modelo...'):
                        forecast = lstm_forecast(df)
                        st.subheader("Previsão para os Próximos 2 Dias")
                        st.dataframe(forecast.style.format({'Previsão': '{:.2f}'}))
                        fig = px.line(df, x='Data', y='Preço', title='Histórico do Preço do Petróleo Brent')
                        fig.add_scatter(x=forecast['Data'], y=forecast['Previsão'], mode='lines+markers', name='Previsão LSTM')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"""
                    ⚠️ Para melhores resultados, recomenda-se prever apenas os dias próximos à última data da base ({last_date.strftime('%d/%m/%Y')}). "
                    "Previsões para datas muito distantes podem ser imprecisas, pois o modelo utiliza apenas os valores anteriores para estimar os próximos."
                """)
                max_days = 90
                data_usuario = st.date_input(
                    "Selecione uma data para previsão:",
                    min_value=last_date + timedelta(days=1),
                    max_value=last_date + timedelta(days=max_days)
                )
                if st.button("Gerar Previsão com Prophet"):
                    with st.spinner('Executando modelo Prophet...'):
                        forecast = prophet_forecast(df, steps=max_days)
                        data_str = pd.to_datetime(data_usuario).strftime('%Y-%m-%d')
                        previsao_data = forecast.loc[forecast['ds'] == data_str, 'yhat']
                        if not previsao_data.empty:
                            valor_previsto = previsao_data.values[0]
                            st.success(f"Previsão para {data_usuario.strftime('%d/%m/%Y')}: US$ {valor_previsto:.2f}")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df['Data'], y=df['Preço'], name='Histórico'))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão Prophet'))
                            fig.add_trace(go.Scatter(
                                x=[data_usuario],
                                y=[valor_previsto],
                                mode='markers',
                                marker=dict(size=12, color='red'),
                                name='Data Selecionada'
                            ))
                            fig.update_layout(title='Histórico e Previsão de Preço do Petróleo Brent')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Data selecionada fora do intervalo de previsão")

elif escolha == "Conclusão":
    st.header("✅ Conclusão")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Modelo LSTM")
        st.write("""
        - Janela temporal de 10 dias e 100 épocas de treino  
        - Previsões recursivas usando os 10 dias mais recentes (reais ou previstos)  
        - Captura padrões recentes e tendências de curto prazo no preço do petróleo Brent
        """)

    with col2:
        st.subheader("Modelo Prophet")
        st.write("""
        - Robusto para identificar tendências de longo prazo e padrões sazonais  
        - Indicado para previsões em datas futuras selecionadas pelo usuário  
        - Recomenda-se focar em horizontes temporais mais curtos para maior precisão
        """)

    with col3:
        st.subheader("Próximos Passos")
        st.write("""
        - Incorporar variáveis externas (produção, estoques, eventos geopolíticos)  
        - Explorar arquiteturas híbridas (LSTM + Prophet) para melhorar a acurácia  
        - Implementar intervalos de confiança e métricas quantitativas (RMSE, MAE, MAPE)  
        - Facilitar a avaliação contínua da performance preditiva
        """)

elif escolha == "Referências":
    st.header("📚 Fontes Científicas e Técnicas")
    st.write("""
    **Estudos sobre Previsão com LSTM**  
    1. [Wang et al. (2024) - Brent Oil Price Prediction Using Bi-LSTM Network](https://arxiv.org/abs/2409.12376)  
    2. [Zhu (2019) - EEMD-LSTM Hybrid Model for Oil Price Forecasting](https://repositorio.utfpr.edu.br/jspui/bitstream/1/35724/1/previsaoprecopetroleo.pdf)  
    3. [BOP-BL Model Experimental Results](https://www.techscience.com/iasc/v26n6/41024/html)  

    **Bibliotecas e Frameworks**  
    4. [PyTorch Documentation - LSTM Module](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)  
    5. [Prophet Forecasting Documentation](https://facebook.github.io/prophet/)  
    6. [NeuralForecast Library for Time Series](https://nixtla.github.io/neuralforecast/)  

    **Fontes de Dados**  
    7. [IPEadata - Série Histórica Brent](http://www.ipeadata.gov.br/ExibeSerie.aspx)  
    8. [U.S. Energy Information Administration](https://www.eia.gov)  

    **Análises de Mercado**  
    9. [Site do ipea](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view  
    10. [OPEC+ Production Decisions Analysis](https://www.opec.org)  
    """)
