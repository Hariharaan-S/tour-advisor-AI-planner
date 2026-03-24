# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from planner import plan_trip, TripPlan

app = FastAPI(title="Trip Advisor API")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


class PlanTripRequest(BaseModel):
    cityName: str
    numberOfDays: int = 4
    budget: Optional[float] = None
    coordinates: Optional[dict] = None
    people: int = 4

@app.post("/plan-trip", response_model=TripPlan)
def plan_trip_api(payload: PlanTripRequest):
    try:
        return plan_trip(payload.cityName, payload.numberOfDays, payload.budget, payload.people, payload.coordinates)
    except Exception as e:
        print("Error in plan_trip_api:", e)
        raise HTTPException(status_code=400, detail=str(e))