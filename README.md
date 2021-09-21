## 🧰 Processamento de Linguagem Natural 📓✒
### [Read in English here](README-en.md).
![version](https://img.shields.io/badge/relise-v1.0.0-important)

Projeto de análise de textos. Repositório destinado a organizar ferramentas e *** disciplina de Processamento de Linguagem Natural da Pós da PUC Minas.

[![License: CC-BY-NC-SA](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

[![NLTK](https://img.shields.io/badge/lib-NLTK-darkorange)](https://www.nltk.org/) [![Psycopg](https://img.shields.io/badge/lib-Psycopg-yellowgreen)](https://www.psycopg.org/) [![SQLalchemy](https://img.shields.io/badge/lib-SQLalchemy-darkred)](https://www.sqlalchemy.org/) [![Pandas](https://img.shields.io/badge/lib-Pandas-white)](https://pandas.pydata.org/) [![NumPy](https://img.shields.io/badge/lib-NumPy-darkblue)](https://numpy.org/)

## Recursos

 - Coleta e cataloga **todos os textos** de um **perfil pessoal**, armazenando os seguintes dados dos textos:
   - id [gerado automaticamente]
   - Título;
   - Conteúdo do texto;
   - URL para texto;
   - Categoria;
   - Data de publicação;
   - Visualizações.
 - Armazena em servidor **PostgreSQL**.

![postgres](images/postgreSQL.jpg)

## Para Adicionar

 - Coleta de outros perfis.

# Executando 🏁

 1. Baixe o webdriver para uso da biblioteca Selenium, disponível [aqui](https://chromedriver.chromium.org/downloads).
 2. Crie um arquivo `.env` na pasta raiz do repositório contendo as seguintes informações:

```python
# Login Recanto das Letras
user =   # Usuário
password =   # Senha

# Webdriver
chrome_driver_path =   # Caminho para Webdriver (p.ex. C:\Users\user\.google\chromedriver.exe)

# PostgreSQL
user_db =   # Usuário do Banco de Dados
password_db =   # Senha
host_db =   # Host (localhost)
port_db =   # Porta
database_db =   # Banco de Dados destino
```
 3. Instale as dependências e execute.

```
$ pip install -r requirements.txt
$ python getpoetry/main.py
```

# Teste 🚧

 - Instale as dependências de desenvolvedor.

```
$ pip install -r requirements-dev.txt
$ pytest getpoetry
```
