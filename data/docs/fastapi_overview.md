# FastAPI Application Creation and Path Operations

Source: https://fastapi.tiangolo.com/tutorial/first-steps/

## Introduction to FastAPI

FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. It is built on top of Starlette for the web parts and Pydantic for the data validation and serialization parts. FastAPI is fully compliant with the ASGI (Asynchronous Server Gateway Interface) standard, allowing high-concurrency asynchronous execution via servers like Uvicorn or Hypercorn.

## Creating a Basic FastAPI Application

To create a basic FastAPI application, instantiate the `FastAPI` class:

```python
from fastapi import FastAPI

app = FastAPI(
    title="Documentation Assistant API",
    version="1.0.0",
    description="A technical assistant for documentation querying."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application"}
```

When running with an ASGI server such as `uvicorn app.main:app --reload`, FastAPI automatically configures routing, serialization, and server event loops.

## Path Operation Functions and HTTP Methods

In FastAPI terminology:
- **Path**: The endpoint URL path, such as `/items` or `/users/{user_id}`.
- **Operation**: The HTTP method used, including `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, `OPTIONS`, and `HEAD`.
- **Path Operation Decorator**: The decorator applied to a Python function to associate it with a path and method, such as `@app.get("/items")` or `@app.post("/items")`.
- **Path Operation Function**: The Python function decorated by the path operation decorator.

FastAPI supports both synchronous (`def`) and asynchronous (`async def`) functions:
- If a function is defined with `async def`, FastAPI runs it directly in the main event loop. This is optimal for non-blocking I/O operations such as asynchronous database queries or HTTP calls.
- If a function is defined with standard `def`, FastAPI runs it in an external thread pool from `anyio`, ensuring that blocking synchronous calls (like file reading or CPU-bound tasks) do not stall the event loop.

## Automatic Interactive Documentation

One of FastAPI's flagship features is automatic API schema generation conforming to the OpenAPI (formerly Swagger) standard. FastAPI automatically exposes two interactive documentation interfaces out of the box:
1. **Swagger UI**: Accessible at `/docs`, offering interactive request testing, schema viewing, and authentication testing.
2. **ReDoc**: Accessible at `/redoc`, offering clean, alternative API documentation layout focused on readability.

The OpenAPI JSON schema is generated dynamically at `/openapi.json`. Titles, descriptions, tags, versions, and server URLs can all be customized within the `FastAPI(...)` constructor.
