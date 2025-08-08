from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:admin@mongo:27017")
DB_NAME = os.getenv("DB_NAME", "vaccine_tracker")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "records")

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]
vaccination_collection = db[COLLECTION_NAME]
hesitancy_collection = db["hesitancy_feedback"]

async def test_connection():
    try:
        await client.server_info()
        print("✅ MongoDB connection successful")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
