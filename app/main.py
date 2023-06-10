from fastapi import FastAPI
from routers import user_router
from routers import statistics_router

app = FastAPI()
app.include_router(user_router.router)
app.include_router(statistics_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)