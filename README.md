## 🤖🧰 Web Scraping | [Recanto das Letras](https://www.recantodasletras.com.br/) 📓✒
### [Read in English here](README-en.md).
![version](https://img.shields.io/badge/relise-v1.0.0-important)

Projeto **Web Scraping** **Python** para o site Recanto das Letras, rede social de compartilhamento de contos, poesias e textos em geral.

_**Web Scraping** **Python** project for the Recanto das Letras website, a social network for sharing stories, poetry and texts in general._

[![License: CC-BY-NC-SA](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

[![Psycopg](https://img.shields.io/badge/lib-Psycopg-yellowgreen)](https://www.psycopg.org/) [![SQLalchemy](https://img.shields.io/badge/lib-SQLalchemy-darkred)](https://www.sqlalchemy.org/) [![Selenium](https://img.shields.io/badge/lib-Selenium-darkgreen)](https://www.selenium.dev/documentation/en/selenium_installation/installing_selenium_libraries/) [![Pandas](https://img.shields.io/badge/lib-Pandas-white)](https://pandas.pydata.org/) [![NumPy](https://img.shields.io/badge/lib-NumPy-darkblue)](https://numpy.org/)

![WebScraping](images/web_scraping.gif)

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
