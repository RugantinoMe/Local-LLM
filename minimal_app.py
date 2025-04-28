from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# Endpoint GET di test
@app.get("/ping")
async def ping():
    print("GET /ping chiamato!")
    return {"message": "Server is running"}

# Endpoint POST di test
@app.post("/test")
async def test_endpoint(request: Request):
    print("POST /test chiamato!")
    try:
        data = await request.json()
        print("Data ricevuta:", data)
    except Exception as e:
        print("Errore durante il parsing:", e)
        return {"error": "Error parsing JSON"}
    return {"message": "Test endpoint reached", "data": data}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
