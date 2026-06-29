from fastapi  import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "name" : "inspection_schedule_ai_assist",
    }