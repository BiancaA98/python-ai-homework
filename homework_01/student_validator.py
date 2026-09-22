import json

from pydantic import BaseModel, EmailStr, PositiveInt


class StudentValidator(BaseModel):
    nume: str
    varsta: PositiveInt
    email: EmailStr


def citeste_student() -> StudentValidator:
    with open("homework_01/student.json", encoding="utf-8") as fisier:
        date = json.load(fisier)
    return StudentValidator.model_validate(date)
