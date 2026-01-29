from fastapi import FastAPI
from routes import register_router
app = FastAPI()

app.include_router(register_router)

# add comment for the command for debugging
# uvicorn main:app --reload