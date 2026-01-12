# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from planner import plan_trip, TripPlan

app = FastAPI(title="Trip Advisor API")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


class PlanTripRequest(BaseModel):
    city: str
    question: str

@app.post("/plan-trip", response_model=TripPlan)
def plan_trip_api(payload: PlanTripRequest):
    try:
        return plan_trip(payload.city, payload.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# @app.post("/plan-trip", response_model=TripPlan)
# def plan_trip_api(city: str, question: str):
#     try:
#         return plan_trip(city, question)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
