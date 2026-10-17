# HTTP Exceptions, Status Codes, and Custom Error Handling

Source: https://fastapi.tiangolo.com/tutorial/handling-errors/

## Raising HTTP Exceptions

To return HTTP error responses to clients, FastAPI provides the `HTTPException` class, which inherits from Starlette's `HTTPException`. When an exception is raised, FastAPI halts execution of the path operation and directly sends the specified error response:

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

items = {"item_1": "Standard Doc"}

@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with identifier '{item_id}' was not found in database.",
            headers={"X-Error-Reason": "ItemNotFound"}
        )
    return {"item": items[item_id]}
```

The resulting response body contains a JSON object with the `"detail"` key:

```json
{
  "detail": "Item with identifier 'unknown' was not found in database."
}
```

## Standard HTTP Status Codes

FastAPI provides standard HTTP status code constants inside `fastapi.status` (which re-exports Starlette status codes) for clean, readable code:
- `status.HTTP_200_OK`: Standard success response.
- `status.HTTP_201_CREATED`: Successful creation of resources.
- `status.HTTP_400_BAD_REQUEST`: Malformed request parameters or business logic invalidity.
- `status.HTTP_401_UNAUTHORIZED`: Authentication failure or missing credentials.
- `status.HTTP_403_FORBIDDEN`: Client authenticated but lacks required permission.
- `status.HTTP_404_NOT_FOUND`: Target resource does not exist.
- `status.HTTP_422_UNPROCESSABLE_ENTITY`: Request payload schema or type validation failure.
- `status.HTTP_500_INTERNAL_SERVER_ERROR`: Unhandled server-side failure.

## Custom Exception Handlers

You can register custom exception handlers for domain-specific exceptions using the `@app.exception_handler()` decorator:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str):
        self.document_id = document_id

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error_code": "DOC_NOT_FOUND",
            "message": f"Documentation with ID {exc.document_id} does not exist.",
            "path": request.url.path
        }
    )
```

## Overriding Default Validation Handlers

You can also override the built-in exception handlers, such as `RequestValidationError` from `fastapi.exceptions`:

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "validation_error",
            "errors": exc.errors(),
            "body": exc.body
        }
    )
```
