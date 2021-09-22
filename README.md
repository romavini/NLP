## 🧰 Processamento de Linguagem Natural 📓✒
### [Read in English here](README-en.md).
![version](https://img.shields.io/badge/relise-v1.0.0-important)

Projeto de análise de textos. Repositório destinado a organizar métodos discutidos na disciplina de Processamento de Linguagem Natural da Pós-graduação da PUCMinas.

[![License: CC-BY-NC-SA](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

[![NLTK](https://img.shields.io/badge/lib-NLTK-FCAF46)](https://www.nltk.org/) [![Psycopg](https://img.shields.io/badge/lib-Psycopg-009977)](https://www.psycopg.org/) [![SQLalchemy](https://img.shields.io/badge/lib-SQLalchemy-d71f00)](https://www.sqlalchemy.org/) [![Pandas](https://img.shields.io/badge/lib-Pandas-130654)](https://pandas.pydata.org/) [![NumPy](https://img.shields.io/badge/lib-NumPy-013243)](https://numpy.org/)

## Recursos

 - Acesso e busca de sentenças em Base de dados;
 - Tolkenização, bag of words e separação de stop-words.

## Para Adicionar

 - Nuvem de palavras;
 - Gerador de textos.

## Executando 🏁

 1. Crie um arquivo `.env` na pasta raiz do repositório contendo as seguintes informações:

```python
# PostgreSQL
user_db =   # Usuário do Banco de Dados
password_db =   # Senha
host_db =   # Host (localhost)
port_db =   # Porta
database_db =   # Banco de Dados destino
```

 2. Instale as dependências e execute.

```
$ pip install -r requirements.txt
$ python nlp/main.py
```

## Teste 🚧

 - Instale as dependências de desenvolvedor.

```
$ pip install -r requirements-dev.txt
$ pytest nlp
```
