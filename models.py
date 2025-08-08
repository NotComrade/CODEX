from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date 

class VaccineRecord(BaseModel):
    disease: str
    vaccine_name: str
    manufacturer: Optional[str] = None
    dose_1_date: str
    dose_2_date: Optional[str] = None
    booster_date: Optional[str] = None

class VaccinationData(BaseModel):
    name: str
    age_group: str
    location: str
    gender: str  # Make sure this is included
    vaccines: List[VaccineRecord]
    timestamp: Optional[datetime] = None