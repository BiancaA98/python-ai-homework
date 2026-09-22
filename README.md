# Teme Python/AI

Acest repository conține temele de la cursul Python/AI. Fiecare temă va fi
păstrată separat, într-un director propriu (`homework_01`, `homework_02` etc.).

Este necesar Python 3.12 sau mai nou și `uv`. Comenzile se rulează din
directorul repository-ului.

## Instalarea dependențelor

```bash
uv sync
```

## Tema 1 — Citire JSON și validare Pydantic

Fișierul `homework_01/student.json` este citit cu modulul standard `json`.
Clasa `StudentValidator` validează câmpurile `nume` (text), `varsta`
(număr întreg pozitiv) și `email` (adresă de email validă).

```bash
uv run python -m homework_01.main
```

## Rularea testelor

```bash
uv run pytest
```

Testele verifică validarea datelor din fișierul JSON real, respingerea unui
email invalid și respingerea vârstelor nepozitive.
