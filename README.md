# Teme Python/AI

Acest repository contine temele de la cursul Python-AI. Fiecare tema va fi
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

## Tema 2 — Agent cu tools

Agentul listeaza fisierele si cauta aparitiile cuvantului `pytest` folosind
comenzile Windows `dir` si `findstr`.

```bash
uv run python -m homework_02.tema2
```

Pentru temele 2 si 3, copiaza `.env.example` in `.env` si completeaza cheia
API si modelul OpenRouter.

## Tema 3 — RAG Search

Adauga fisierele tale TXT, MD sau PDF direct in `homework_03/data_sources`.
Folderul este initial gol; documentele si baza locala nu se includ in Git.
PDF-urile trebuie sa contina text (nu este inclus OCR).

Documentele sunt impartite in chunk-uri cu suprapunere, transformate in
embeddings si salvate in ChromaDB. Fiecare chunk pastreaza numele fisierului
in `source` si pagina pentru PDF; textele simple nu au numar de pagina.
Repetarea ingestiei nu adauga duplicate si actualizeaza chunk-urile modificate.

```bash
uv run python -m homework_03.tema3 ingest
uv run python -m homework_03.tema3 search
```

La cautare introduci numele exact al fisierului si intrebarea. Sunt cautate
doar chunk-urile acelui fisier, cu distanta cosine cel mult `0.5`.
LLM-ul primeste cel mult 4 chunk-uri si raspunde din context, cu referinte
la fisier si pagini. Fara rezultate relevante nu se apeleaza LLM-ul.

Optional, `--rerank` reordoneaza pana la 15 rezultate folosind cross-encoder-ul
usor TinyBERT, dupa filtrarea distantei. Pastreaza primele 4 rezultate cu scor
cel putin `--min-relevance` (scor intre 0 si 1, nu o probabilitate garantata).
Pragurile pot fi ajustate:

```bash
uv run python -m homework_03.tema3 search --rerank --max-distance 0.5 --min-relevance 0.5
```

Modelele se descarca la prima utilizare si ruleaza local pe CPU. Modelele
alese sunt pentru engleza; pentru alte limbi calitatea poate varia.
Strategia urmeaza [documentatia Sentence Transformers](https://www.sbert.net/docs/cross_encoder/pretrained_models.html).

## Rularea testelor

```bash
uv run pytest
```

Testele verifica validarea datelor din fisierul JSON real, respingerea unui
email invalid si respingerea unei varste egale cu zero sau negative.
Pentru Tema 3 sunt verificate ingestia fara duplicate, filtrarea pe fisier,
pragurile, re-ranking-ul si evitarea apelului LLM fara rezultate.
