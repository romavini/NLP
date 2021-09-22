## 🧰 Natural Language Processing 📓✒
### [Leia em Português aqui](README.md).
![version](https://img.shields.io/badge/relise-v1.0.0-important)

Text analysis project. Repository designed to organize methods discussed in the Natural Language Processing course at PUCMinas's Postgraduating.

[![License: CC-BY-NC-SA](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

[![NLTK](https://img.shields.io/badge/lib-NLTK-FCAF46)](https://www.nltk.org/) [![Psycopg](https://img.shields.io/badge/lib-Psycopg-009977)](https://www.psycopg.org/) [![SQLalchemy](https://img.shields.io/badge/lib-SQLalchemy-d71f00)](https://www.sqlalchemy.org/) [![Pandas](https://img.shields.io/badge/lib-Pandas-130654)](https://pandas.pydata.org/) [![NumPy](https://img.shields.io/badge/lib-NumPy-013243)](https://numpy.org/)

## Resources

 - Access and search of sentences in Database;
 - Tolkenization, bag of words and stop-words separation.

## To add

 - Word cloud;
 - Text generator.

## Running 🏁

 1. Create a .env file in the repository root folder containing the following information:

```python
# PostgreSQL
user_db =   # User in database
password_db =   # Password
host_db =   # Host (localhost)
port_db =   # Port
database_db =   # Database
```

 2. Install dependencies and run.

```
$ pip install -r requirements.txt
$ python nlp/main.py
```

## Test 🚧

 - Install developer dependencies.

```
$ pip install -r requirements-dev.txt
$ pytest nlp
```
