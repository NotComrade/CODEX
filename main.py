# main.py
from fastapi import FastAPI, Request, HTTPException, Query, Body, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from bson import ObjectId
from dotenv import load_dotenv
import pandas as pd
import io
import os
import time
import asyncio
from collections import defaultdict, deque

# Import modules from your project structure
from models import VaccinationData, VaccineRecord # Ensure these are correctly imported
from database import vaccination_collection, hesitancy_collection # Ensure these are correctly imported
from utils import anonymize_data # Ensure this is correctly imported
from disease_data import DISEASE_VACCINE_MAP # Ensure this is correctly imported

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Community Vaccination Tracker")

# Configure Jinja2Templates for serving HTML
templates = Jinja2Templates(directory=".") # Assuming frontend.html is in the same directory

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT: Restrict this in production! e.g., ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory rate limiting tracker
ip_request_log = defaultdict(lambda: deque())

RATE_LIMIT = 3          # Max 3 requests
TIME_WINDOW = 10        # Within 10 seconds
CLEANUP_INTERVAL = 60   # Clean every 60 seconds

@app.on_event("startup")
async def startup_db_client():
    from database import test_connection
    await test_connection()


# Background cleanup task for rate limit logs
@app.on_event("startup")
async def start_cleanup_task():
    async def cleanup():
        while True:
            current_time = time.time()
            for ip in list(ip_request_log.keys()):
                ip_request_log[ip] = deque([
                    t for t in ip_request_log[ip]
                    if current_time - t <= TIME_WINDOW
                ])
                if not ip_request_log[ip]:
                    del ip_request_log[ip]
            await asyncio.sleep(CLEANUP_INTERVAL)

    asyncio.create_task(cleanup())

# --- Frontend Serving Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """
    Serves the main frontend HTML page.
    """
    return templates.TemplateResponse("frontend.html", {"request": request})


# --- Backend API Endpoints (using MongoDB) ---

# Submit vaccination record
@app.post("/submit")
async def submit_data(data: VaccinationData, request: Request):
    try:
        data_dict = data.dict()
        data_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Insert into MongoDB
        result = await vaccination_collection.insert_one(data_dict)
        
        if result.inserted_id:
            return JSONResponse(
                status_code=201,
                content={
                    "message": "Data submitted successfully",
                    "id": str(result.inserted_id)
                }
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Data insertion failed: {str(e)}"
        )

# Get all records with optional date range
@app.get("/records")
async def get_all_records(start: Optional[str] = None, end: Optional[str] = None):
    query = {}
    if start and end:
        try:
            # Ensure timestamps are in ISO format for comparison
            query["timestamp"] = {
                "$gte": datetime.fromisoformat(start).isoformat(),
                "$lte": datetime.fromisoformat(end).isoformat()
            }
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 (e.g., 2023-10-27T10:00:00).")
    records = []
    async for doc in vaccination_collection.find(query):
        doc["_id"] = str(doc["_id"])
        records.append(doc)
    return records

# Get today's records
@app.get("/records/today")
async def get_today_records():
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59)
    query = {"timestamp": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
    records = []
    async for doc in vaccination_collection.find(query):
        doc["_id"] = str(doc["_id"])
        records.append(doc)
    return records

# Export records to CSV
@app.get("/records/export")
async def export_records(start: Optional[str] = None, end: Optional[str] = None):
    query = {}
    if start and end:
        try:
            query["timestamp"] = {
                "$gte": datetime.fromisoformat(start).isoformat(),
                "$lte": datetime.fromisoformat(end).isoformat()
            }
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 (e.g., 2023-10-27T10:00:00).")
    docs = []
    async for doc in vaccination_collection.find(query):
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    if not docs:
        raise HTTPException(status_code=404, detail="No records found for the given criteria.")
    df = pd.DataFrame(docs)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=vaccination_records.csv"
    })

# Stats by disease
@app.get("/vaccines/stats")
async def disease_stats(disease: str):
    # Count documents where any vaccine in the 'vaccines' array has the specified disease
    count = await vaccination_collection.count_documents({"vaccines.disease": disease})
    return {"disease": disease, "total_records": count}

# Group stats by field
@app.get("/vaccines/stats/groupby")
async def group_by_stats(field: str = Query(..., pattern="^(age_group|location)$")):
    pipeline = [{"$group": {"_id": f"${field}", "count": {"$sum": 1}}}]
    results = await vaccination_collection.aggregate(pipeline).to_list(None)
    return [{"field": doc["_id"], "count": doc["count"]} for doc in results]

# Map clusters (using actual location data from DB)
@app.get("/map/clusters")
async def map_clusters():
    # Aggregate by location and count, then fetch coordinates (if available)
    pipeline = [
        {"$group": {"_id": "$location", "count": {"$sum": 1}}}
    ]
    results = await vaccination_collection.aggregate(pipeline).to_list(None)

    # For demonstration, we'll use dummy coordinates or a very simple lookup.
    # In a real app, you'd need a geo-coding service or pre-defined coordinates for locations.
    # For now, let's just return the location and count. Frontend will handle dummy coords.
    return [{"location": doc["_id"], "count": doc["count"]} for doc in results]


# Disease data
@app.get("/diseases/all")
def all_diseases():
    return DISEASE_VACCINE_MAP

@app.get("/diseases/search")
def search_diseases(q: str):
    return {"matches": [d for d in DISEASE_VACCINE_MAP if d.lower().startswith(q.lower())]}

# Admin delete
class AdminDeleteRequest(BaseModel):
    password: str

@app.delete("/delete/{record_id}")
async def delete_record(record_id: str, request: AdminDeleteRequest = Body(...)):
    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_password:
        raise HTTPException(status_code=500, detail="Admin password not configured on server")

    if request.password != admin_password:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid admin password")

    try:
        result = await vaccination_collection.delete_one({"_id": ObjectId(record_id)})
    except Exception: # Catch potential invalid ObjectId format
        raise HTTPException(status_code=400, detail="Invalid record ID format.")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Record not found")

    return {"message": "Record deleted successfully"}

# Hesitancy feedback model (defined here for clarity, but could be in models.py)
class HesitancyEntry(BaseModel):
    age_group: str
    reason: str
    timestamp: datetime

# Hesitancy endpoints
@app.post("/submit-hesitancy")
async def submit_hesitancy(age_group: str = Form(...), reason: str = Form(...)):
    if not age_group or not reason.strip():
        raise HTTPException(status_code=400, detail="Please select your age group and provide a reason.")

    entry = {
        "age_group": age_group,
        "reason": reason.strip(),
        "timestamp": datetime.utcnow().isoformat()
    }

    await hesitancy_collection.insert_one(entry)

    return {"message": "Thank you for your feedback. Your response has been recorded anonymously."}

@app.get("/hesitancy-responses")
async def get_hesitancy_responses():
    responses = []
    docs = await hesitancy_collection.find().to_list(length=1000)
    for doc in docs:
        doc["_id"] = str(doc["_id"]) # Convert ObjectId to string
        responses.append(doc)
    return responses

# --- Dashboard Data API (Calculated from DB) ---
@app.get("/api/dashboard-data")
async def get_dashboard_data():
    total_records = await vaccination_collection.count_documents({})
    
    # Count fully vaccinated (at least 2 doses or booster)
    fully_vaccinated_count = await vaccination_collection.count_documents({
        "vaccines": {
            "$elemMatch": {
                "$or": [
                    {"dose_2_date": {"$ne": None}},
                    {"booster_date": {"$ne": None}}
                ]
            }
        }
    })

    # Count partially vaccinated (at least 1 dose, but not fully)
    partially_vaccinated_count = await vaccination_collection.count_documents({
        "vaccines": {
            "$elemMatch": {
                "dose_1_date": {"$ne": None},
                "dose_2_date": None,
                "booster_date": None
            }
        }
    })

    fully_vaccinated_pct = (fully_vaccinated_count / total_records * 100) if total_records > 0 else 0
    partially_vaccinated_pct = (partially_vaccinated_count / total_records * 100) if total_records > 0 else 0
    not_vaccinated_pct = 100 - fully_vaccinated_pct - partially_vaccinated_pct

    # Get today's records count
    records_today_count = len(await get_today_records())
    
    return {
        "fully_vaccinated": {"percentage": round(fully_vaccinated_pct, 1), "change": 8, "trend": "increase"},
        "partially_vaccinated": {"percentage": round(partially_vaccinated_pct, 1), "change": 3, "trend": "decrease"},
        "not_vaccinated": {"percentage": round(not_vaccinated_pct, 1), "change": 5, "trend": "decrease"},
        "records_today": {"count": records_today_count, "change": 112, "period": "this month"}
    }

# --- Report Data API (Calculated from DB) ---
@app.get("/api/report-data")
async def get_report_data():
    """
    Endpoint to generate comprehensive vaccination report data for dashboard visualizations.
    Returns statistics grouped by age, gender, literacy (simulated), and time trends.
    """
    
    # --- AGE GROUP DATA ---
    # Calculate vaccination rates by age group
    age_group_pipeline = [
        {"$unwind": "$vaccines"},  # Split each vaccine record into separate documents
        {"$group": {
            "_id": {
                "age_group": "$age_group",  # Group by age group
                "is_vaccinated": {"$cond": [{"$ne": ["$vaccines.dose_1_date", None]}, True, False]}  # Check if vaccinated
            },
            "count": {"$sum": 1}  # Count records in each group
        }}
    ]
    age_group_raw_data = await vaccination_collection.aggregate(age_group_pipeline).to_list(None)

    # Process raw age group data into vaccinated/not vaccinated counts
    age_groups_processed = defaultdict(lambda: {"vaccinated": 0, "not_vaccinated": 0})
    for item in age_group_raw_data:
        age_group = item["_id"]["age_group"]
        is_vaccinated = item["_id"]["is_vaccinated"]
        count = item["count"]
        if is_vaccinated:
            age_groups_processed[age_group]["vaccinated"] += count
        else:
            age_groups_processed[age_group]["not_vaccinated"] += count

    # Calculate percentages for each age group
    age_group_labels = sorted(list(age_groups_processed.keys()))
    age_group_vaccinated_pct = []
    age_group_not_vaccinated_pct = []
    for label in age_group_labels:
        total = age_groups_processed[label]["vaccinated"] + age_groups_processed[label]["not_vaccinated"]
        age_group_vaccinated_pct.append(
            round((age_groups_processed[label]["vaccinated"] / total * 100), 1) if total > 0 else 0
        )
        age_group_not_vaccinated_pct.append(
            round((age_groups_processed[label]["not_vaccinated"] / total * 100), 1) if total > 0 else 0
        )

    # --- GENDER DATA ---
    # Calculate distribution of records by gender
    gender_pipeline = [
        {"$group": {
            "_id": "$gender",  # Group by gender field
            "count": {"$sum": 1}  # Count records in each gender group
        }}
    ]
    gender_raw_data = await vaccination_collection.aggregate(gender_pipeline).to_list(None)
    
    # Process gender data into percentages
    gender_labels = [doc["_id"] for doc in gender_raw_data]
    gender_counts = [doc["count"] for doc in gender_raw_data]
    total_genders = sum(gender_counts)
    gender_data_pct = [
        round((count / total_genders * 100), 1) if total_genders > 0 else 0 
        for count in gender_counts
    ]

    # --- LITERACY DATA (SIMULATED) ---
    # Note: Actual literacy data would require a 'literacy' field in the documents
    # Currently simulating based on vaccination status
    
    # Get total vaccinated and unvaccinated counts
    total_vaccinated = await vaccination_collection.count_documents({"vaccines.dose_1_date": {"$ne": None}})
    total_not_vaccinated = await vaccination_collection.count_documents({"vaccines.dose_1_date": None})
    total_population = total_vaccinated + total_not_vaccinated

    # Simulate literacy distribution (70% of vaccinated are literate)
    literate_vaccinated_pct = round((total_vaccinated / total_population * 70), 1) if total_population > 0 else 0
    illiterate_vaccinated_pct = round((total_vaccinated / total_population * 30), 1) if total_population > 0 else 0

    # Simulate literacy distribution (40% of unvaccinated are literate)
    literate_not_vaccinated_pct = round((total_not_vaccinated / total_population * 40), 1) if total_population > 0 else 0
    illiterate_not_vaccinated_pct = round((total_not_vaccinated / total_population * 60), 1) if total_population > 0 else 0

    # Prepare literacy data for chart
    literacy_vaccinated_data = [literate_vaccinated_pct, illiterate_vaccinated_pct]
    literacy_not_vaccinated_data = [literate_not_vaccinated_pct, illiterate_not_vaccinated_pct]

    # --- TREND DATA (SIMULATED) ---
    # Note: Actual trend data would require historical records with timestamps
    # Currently using hardcoded values for demonstration
    
    # Monthly trend labels (last 10 months)
    trend_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    
    # Simulated vaccination trend data
    trend_fully_vaccinated = [42, 48, 53, 56, 59, 62, 66, 68, 71, 72]  # % fully vaccinated
    trend_partially_vaccinated = [23, 20, 18, 16, 18, 17, 16, 16, 15, 15]  # % partially vaccinated
    trend_not_vaccinated = [35, 32, 29, 28, 23, 21, 18, 16, 14, 13]  # % not vaccinated

    # --- FINAL RESPONSE STRUCTURE ---
    return {
        "age_group_data": {
            "labels": age_group_labels,
            "vaccinated": age_group_vaccinated_pct,
            "not_vaccinated": age_group_not_vaccinated_pct
        },
        "gender_data": {
            "labels": gender_labels,
            "data": gender_data_pct  # Gender distribution percentages
        },
        "literacy_data": {
            "labels": ['Literate', 'Illiterate'],
            "vaccinated": literacy_vaccinated_data,  # Literacy among vaccinated
            "not_vaccinated": literacy_not_vaccinated_data  # Literacy among unvaccinated
        },
        "trend_data": {
            "labels": trend_labels,
            "fully_vaccinated": trend_fully_vaccinated,  # Monthly fully vaccinated %
            "partially_vaccinated": trend_partially_vaccinated,  # Monthly partially vaccinated %
            "not_vaccinated": trend_not_vaccinated  # Monthly unvaccinated %
        }
    }

# API endpoint to provide current vaccine stock updates (using dummy data for now)
@app.get("/api/vaccine-stock")
async def get_vaccine_stock():
    # In a real application, this would fetch from a 'stock' collection in MongoDB
    # For now, using a hardcoded list as there's no stock collection in your DB setup
    return [
        {"type": "Covishield", "doses": 100, "date": "Aug 7, 2025"},
        {"type": "Moderna", "doses": 200, "date": "Aug 6, 2025"},
        {"type": "Covaxin", "doses": 150, "date": "Aug 5, 2025"},
    ]

