# Teme Python/AI

Acest repository contine temele de la cursul Python/AI. Fiecare tema va fi
pastrata separat, intr-un director propriu (`homework_01`, `homework_02` etc.).

Este necesar Python 3.12 sau mai nou si `uv`. Comenzile se ruleaza din
directorul repository-ului.

## Instalarea dependentelor

```bash
uv sync
```

## Tema 1 — Citire JSON si validare Pydantic

Fisierul `homework_01/student.json` este citit cu modulul standard `json`.
Clasa `StudentValidator` valideaza campurile `nume` (text), `varsta`
(numar intreg pozitiv) si `email` (adresa de email valida).

```bash
uv run python -m homework_01.main
```

## Rularea testelor

```bash
uv run pytest
```

Testele verifica validarea datelor din fisierul JSON real, respingerea unui
email invalid si respingerea varstelor nepozitive.
