import uvicorn
from utils.Config import config

if __name__ == "__main__":
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=int(config.port),  # Ensure port is an integer
        reload=True  # Used only in development mode
    )