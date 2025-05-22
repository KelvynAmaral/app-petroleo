# ⛽ Análise e Previsão do Preço do Petróleo Brent

Este repositório contém uma aplicação interativa para análise e previsão do preço do petróleo Brent. Utilizando modelos de Machine e Deep Learning  — **Prophet** e **LSTM** — a aplicação gera previsões e oferece insights por meio de um painel construído com Streamlit. A aplicação pode ser executada localmente, via Docker ou ser publicada diretamente na nuvem. ☁️

---

## 🔍 Funcionalidades

-   Análise de dados históricos do petróleo Brent. 📊
-   Geração de insights sobre o mercado. 💡
-   Previsão de preços futuros com Prophet e LSTM. 📈
-   Visualizações e interações via Streamlit. 🖥️
-   Suporte a execução com Docker. 🐳
-   Deploy simples com Streamlit Cloud (com ou sem Docker). 🚀

---

## 🛠️ Tecnologias Utilizadas

-   **Python** – linguagem principal do projeto. 🐍
-   **Prophet** – modelo estatístico do Facebook para séries temporais. 🕰️
-   **LSTM (Long Short-Term Memory)** – rede neural recorrente para previsão. 🧠
-   **TensorFlow/Keras** – frameworks para o modelo LSTM. 🔥
-   **Pandas e NumPy** – manipulação de dados e cálculos numéricos. 🐼🔢
-   **Matplotlib/Seaborn** – visualizações complementares. 🎨
-   **Streamlit** – criação da interface interativa. ✨
-   **Docker** – conteinerização e execução isolada. 📦

---

## 📁 Estrutura do Repositório

-   `Dashboard.py`: script principal com a interface, análise e previsões. 📝
-   `Dockerfile`: instruções para construção da imagem Docker. 🐳
-   `requirements.txt`: lista de dependências do Python. 📄
-   `streamlit.yaml`: configurações opcionais para deploy no Streamlit Cloud. ⚙️

---

## ✅ Pré-requisitos

Antes de iniciar, certifique-se de ter instalado:

-   Python 3.8 ou superior 🐍
-   Git 🔧
-   Docker (opcional, se quiser usar containers) 🐳

---

## 🚀 Instalação Local

1.  Clone o repositório:

    ```
    git clone https://github.com/Data-Analitycs-Pos-Tech-Fiap/app-petroleo.git
    cd app-petroleo
    ```

2.  Crie e ative um ambiente virtual:

    ```
    python -m venv venv
    ```

    *   Linux/macOS:

        ```
        source venv/bin/activate
        ```

    *   Windows:

        ```
        venv\Scripts\activate
        ```

3.  Instale as dependências:

    ```
    pip install -r requirements.txt
    ```

4.  Execute a aplicação:

    ```
    streamlit run Dashboard.py
    ```

    Acesse no navegador: [http://localhost:8501](http://localhost:8501) 🌐

---

## 🐳 Execução com Docker

1.  Construa a imagem:

    ```
    docker build -t brent-analysis-app .
    ```

2.  Execute o container:

    ```
    docker run -p 8501:8501 brent-analysis-app
    ```

3.  Acesse a aplicação em: [http://localhost:8501](http://localhost:8501) 🌐

---

## ☁️ Deploy com Streamlit Cloud

1.  Acesse: [https://streamlit.io/cloud](https://streamlit.io/cloud)
2.  Faça login com sua conta GitHub.
3.  Crie um novo app apontando para o repositório.
4.  Defina `Dashboard.py` como o arquivo principal.
5.  Clique em **Deploy**. 🎉

---

## 🤝 Contribuindo

Contribuições são muito bem-vindas! 🙌
Você pode abrir *issues* para reportar problemas ou sugerir melhorias. 🐞💡
Se quiser contribuir com código:

1.  Faça um *fork* deste repositório. 🍴
2.  Crie uma nova *branch* com sua funcionalidade (`git checkout -b feature/sua-feature`).
3.  Faça *commit* das suas alterações (`git commit -m 'Adiciona nova feature'`).
4.  Envie para a *branch* original (`git push origin feature/sua-feature`).
5.  Abra um *Pull Request*. 🔄

---

## 👤 Autor

**Kelvyn Amaral** 👨‍💻

---

**Nota:**
Certifique-se de substituir quaisquer URLs ou caminhos específicos conforme necessário para o seu ambiente. 📌
