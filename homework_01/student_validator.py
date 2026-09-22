import json
from pathlib import Path

from pydantic import BaseModel, EmailStr, PositiveInt


class StudentValidator(BaseModel):
    nume: str
    varsta: PositiveInt
    email: EmailStr


def citeste_student() -> StudentValidator:
    cale = Path(__file__).with_name("student.json")
    with cale.open(encoding="utf-8") as fisier:
        date = json.load(fisier)
    return StudentValidator.model_validate(date)


if __name__ == "__main__":
    print(citeste_student())
