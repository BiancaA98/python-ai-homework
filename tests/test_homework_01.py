import pytest
from pydantic import ValidationError

from homework_01.student_validator import StudentValidator, citeste_student


def test_validare_student():
    student = citeste_student()

    assert student.nume == "Ana Popescu"
    assert student.varsta == 20
    assert student.email == "ana.popescu@example.com"


def test_email_invalid():
    date = citeste_student().model_dump()
    date["email"] = "email-invalid"

    with pytest.raises(ValidationError) as eroare:
        StudentValidator.model_validate(date)

    assert eroare.value.errors()[0]["loc"] == ("email",)


@pytest.mark.parametrize("varsta", [0, -1])
def test_varsta_nepozitiva(varsta):
    date = citeste_student().model_dump()
    date["varsta"] = varsta

    with pytest.raises(ValidationError) as eroare:
        StudentValidator.model_validate(date)

    assert eroare.value.errors()[0]["loc"] == ("varsta",)
