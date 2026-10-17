# FastAPI Routing, Path Parameters, and Query Parameters

Source: https://fastapi.tiangolo.com/tutorial/path-params/

## Modular Routing with APIRouter

In non-trivial applications, keeping all path operations in a single file becomes unmaintainable. FastAPI provides `APIRouter` to structure endpoints modularly into distinct components, submodules, or domains:

```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Document not found"}}
)

@router.get("/")
def list_documents():
    return [{"id": 1, "title": "Overview"}]
```

Include the router in the root application using `app.include_router(router)`. Routers support their own dependencies, prefixes, tags, and lifecycle events.

## Path Parameters and Type Conversion

Path parameters allow capturing dynamic values directly from the URL. Declaring Python type annotations triggers automatic parsing, validation, and serialization:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

If a client sends `/items/foo`, FastAPI detects that `foo` cannot be converted to `int` and immediately responds with an HTTP 422 Unprocessable Entity error detailing the exact validation failure, without executing the function body.

### Path Parameter Ordering Considerations

Path operations are evaluated sequentially in the order they are registered. Specific fixed paths must be declared before parameterized paths to prevent collisions:

```python
# Fixed path must be declared first
@app.get("/users/me")
def read_current_user():
    return {"user_id": "current_user"}

# Parameterized path declared second
@app.get("/users/{user_id}")
def read_user(user_id: str):
    return {"user_id": user_id}
```

## Query Parameters and Validation

Function parameters that are not matched in the path template are automatically interpreted as URL query parameters. Query parameters can be required, optional, or have default values:

```python
from typing import Optional
from fastapi import Query

@app.get("/search")
def search_documents(
    q: str = Query(..., min_length=3, max_length=100, description="Search query string"),
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    filter_tag: Optional[str] = Query(None, alias="tag")
):
    return {"q": q, "skip": skip, "limit": limit, "tag": filter_tag}
```

FastAPI's `Query` class provides rich metadata and validation capabilities including minimum/maximum lengths, regex patterns, numeric ranges, aliases, deprecation flags, and OpenAPI documentation descriptions.
