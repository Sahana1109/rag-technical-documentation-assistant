# FastAPI Dependency Injection System

Source: https://fastapi.tiangolo.com/tutorial/dependencies/

## Understanding Dependency Injection in FastAPI

FastAPI features an extremely powerful yet intuitive Dependency Injection (DI) system built around the `Depends` class. Dependency injection allows declaring shared logic, database sessions, authentication guards, security policies, and configuration loaders without manual instantiation or boilerplate.

Key advantages of FastAPI's DI system include:
- Shared code across path operations without duplication.
- Automatic parameter resolution and validation inside dependencies.
- Painless testing through dependency overrides (`app.dependency_overrides`).
- Hierarchical dependency resolution (dependencies can depend on other dependencies).

## Basic Dependency Usage

To declare a dependency, create a standard function and inject it using `Depends()`:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def common_pagination_params(skip: int = 0, limit: int = 20):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
def read_items(pagination: dict = Depends(common_pagination_params)):
    return pagination

@app.get("/users/")
def read_users(pagination: dict = Depends(common_pagination_params)):
    return pagination
```

FastAPI evaluates the parameters of `common_pagination_params`, extracts `skip` and `limit` from the request query string, calls the dependency, and passes its return value to the path operation function.

## Hierarchical and Sub-Dependencies

Dependencies can be nested indefinitely. A dependency function can declare its own dependencies via `Depends(...)`:

```python
def get_db_connection():
    return DatabaseSession()

def get_repository(db = Depends(get_db_connection)):
    return DocumentRepository(db=db)

@app.get("/documents/")
def list_docs(repo: DocumentRepository = Depends(get_repository)):
    return repo.get_all()
```

FastAPI builds a dependency graph, executes each node in topological order, and caches results within a single request by default (`use_cache=True`).

## Dependencies with `yield` (Cleanup and Context Management)

For resources requiring teardown (such as database sessions, thread locks, or file handles), FastAPI supports `yield` dependencies:

```python
from typing import Generator

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/records/")
def get_records(db: SessionLocal = Depends(get_db)):
    return db.query(Record).all()
```

The code before `yield` executes prior to running the path operation. The code after `yield` executes after the response is generated, even if an exception occurs during the request.

## Global and Router-Level Dependencies

Dependencies can be applied globally to entire routers or the entire application:

```python
router = APIRouter(dependencies=[Depends(verify_token), Depends(enforce_rate_limit)])
app = FastAPI(dependencies=[Depends(global_audit_logger)])
```
