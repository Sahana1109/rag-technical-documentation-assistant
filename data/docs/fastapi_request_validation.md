# Request Validation and Response Models with Pydantic

Source: https://fastapi.tiangolo.com/tutorial/body/

## Pydantic Models for Request Bodies

FastAPI leverages Pydantic for fast, robust data validation, type conversion, and serialization. Request bodies are declared as subclasses of `pydantic.BaseModel`:

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique account username")
    email: EmailStr = Field(..., description="User's primary contact email")
    age: Optional[int] = Field(None, ge=18, le=120, description="Age must be at least 18")
    is_active: bool = Field(default=True)
```

When this schema is used in a path operation parameter:

```python
@app.post("/users/")
def create_user(user: UserCreate):
    return user
```

FastAPI reads the JSON request payload, validates every field against the Pydantic model specifications, handles conversions (e.g. string "19" to integer 19), and instantiates the `UserCreate` instance.

## Error Handling on Validation Failures

If incoming data fails validation (such as a missing required field or an invalid email format), FastAPI catches the Pydantic `ValidationError` and responds with an HTTP status code `422 Unprocessable Entity`:

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "username"],
      "msg": "String should have at least 3 characters",
      "input": "ab",
      "ctx": {"min_length": 3}
    }
  ]
}
```

The error details include the exact path to the invalid field, the error type, and a descriptive error message.

## Response Models and Data Filtering

FastAPI allows defining a `response_model` on path operation decorators to control output serialization:

```python
class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool

@app.post("/users/", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate):
    # Password and internal attributes are omitted from UserResponse
    db_user = db.save(user)
    return db_user
```

Benefits of `response_model`:
- **Security**: Strips sensitive fields (such as hashed passwords or internal database flags) before sending output to clients.
- **Documentation**: Automatically documents the response schema in Swagger UI and OpenAPI specifications.
- **Conversion**: Validates and serializes output dictionaries or ORM models into clean JSON schemas.
- **Flag Controls**: Supports `response_model_exclude_unset=True` to omit fields that were not explicitly set.
