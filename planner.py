# planner.py
from typing import List, Union
from pydantic import BaseModel
import json
import ollama
import re
from pymongo import MongoClient

# =============================
# MongoDB Client Configuration
# =============================

client = MongoClient("mongodb://localhost:27017")
db = client["tour_advisor"]

places_col = db["places"]
transport_col = db["transport_routes"]


# ===============================
# 📦 Models
# ===============================

class TouristSpot(BaseModel):
    name: str
    popularity: Union[float, str]
    description: str

class Transport(BaseModel):
    name: str
    average_cost: Union[int, str]
    duration: Union[int, str]

class InstructionStep(BaseModel):
    day: int
    time: str
    place_name: str
    location_link: str

class CostSummary(BaseModel):
    total_cost_for_people: Union[int, str]
    people_count: int

class TripPlan(BaseModel):
    title: str
    description: str
    tourist_spots: List[TouristSpot]
    transport: List[Transport]
    instructions: List[InstructionStep]
    cost_summary: CostSummary
    people: int

# ===============================
# 🛠 Helpers (Deterministic)
# ===============================

def extract_days(question: str, default=1):
    match = re.search(r"(\d+)\s*(day|days)", question.lower())
    if match:
        return int(match.group(1))
    return default


def fetch_places_by_city(city: str):
    return list(
        places_col.find(
            {"city": {"$regex": f"^{city}$", "$options": "i"}},
            {"_id": 0}
        )

    )

def extract_people(question: str, default=4):
    match = re.search(
        r"(for|of)\s+(\d+)\s+(people|persons|members)",
        question.lower()
    )
    if match:
        return int(match.group(2))
    return default

def select_top_k_places(places, k):
    return sorted(
        places,
        key=lambda x: x.get("popularity", 0),
        reverse=True
    )[:k]


def get_transport(src_id, dest_id):
    routes = list(
        transport_col.find(
            {
                "src_place_id": src_id,
                "dest_place_id": dest_id
            },
            {"_id": 0}
        )
    )

    return sorted(routes, key=lambda x: x["cost"])


def minutes_to_time_str(minutes_from_midnight: int) -> str:
    hours = minutes_from_midnight // 60
    minutes = minutes_from_midnight % 60
    suffix = "AM" if hours < 12 else "PM"
    hours = hours if hours <= 12 else hours - 12
    return f"{hours:02d}:{minutes:02d} {suffix}"


def build_instruction_steps(day_groups) -> List[InstructionStep]:
    instructions = []

    for day_index, day_places in enumerate(day_groups, start=1):
        current_time = int(DAY_START_MIN)  # 9:30 AM

        for i, place in enumerate(day_places):
            # Visit time
            visit_duration = estimate_visit_duration(place)
            visit_time_str = minutes_to_time_str(current_time)

            instructions.append(
                InstructionStep(
                    day=day_index,
                    time=visit_time_str,
                    place_name=place["name"],
                    location_link=place.get("google_maps_link", "NOT_AVAILABLE")
                )
            )

            current_time += visit_duration

            # Add travel time AFTER visit (if next place exists)
            if i < len(day_places) - 1:
                routes = get_transport(
                    place["place_id"],
                    day_places[i + 1]["place_id"]
                )
                if routes:
                    travel_duration = routes[0].get("duration")
                    if isinstance(travel_duration, int):
                        current_time += travel_duration


    return instructions

# ===============================
# ⏱ Visit Duration Estimator (minutes)
# ===============================

def estimate_visit_duration(place):
    name = place["name"].lower()
    desc = place.get("description", "").lower()

    if "temple" in name or "temple" in desc:
        return 60
    if "beach" in name:
        return 90
    if "museum" in name:
        return 120
    if "park" in name:
        return 75

    return 60  # safe default


# ===============================
# 📅 Day-wise Planner (9:30 AM – 6:00 PM)
# ===============================

DAY_START_MIN = 9 * 60 + 30  # 570 (int)   # 9:30 AM
DAY_END_MIN = 18 * 60     # 6:00 PM
DAY_LIMIT = DAY_END_MIN - DAY_START_MIN  # 510 minutes


def split_into_days(places):
    days = []
    current_day = []
    time_used = 0

    for i, place in enumerate(places):
        visit_time = estimate_visit_duration(place)
        travel_time = 0

        if current_day:
            prev = current_day[-1]
            routes = get_transport(prev["place_id"], place["place_id"])
            if routes:
                travel_time = routes[0]["duration"]

        if time_used + visit_time + travel_time > DAY_LIMIT:
            days.append(current_day)
            current_day = []
            time_used = 0

        current_day.append(place)
        time_used += visit_time + travel_time

    if current_day:
        days.append(current_day)

    return days
# ===============================
# 🤖 LLM — Narration ONLY
# ===============================

SYSTEM_PROMPT = """
You are an AI Trip Narrator.

RULES:
- You do NOT plan routes
- You ONLY narrate provided route facts
- Days start at 9:30 AM and end at 6:00 PM
- Narrate day by day
- One sentence per instruction
- Do NOT invent places, time slots, or transport
- If something is missing, say NOT_AVAILABLE
"""


def ask_llm_to_plan(places, days, city, question):
    """
    LLM decides WHICH places to visit and in WHAT order.
    It does NOT calculate time or cost.
    """

    place_summaries = [
        {
            "place_id": p["place_id"],
            "name": p["name"],
            "popularity": p.get("popularity", 0),
            "category": p.get("category", "NOT_AVAILABLE")
        }
        for p in places
    ]

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": """
You are an AI Trip Planner.

RULES:
- Use ONLY the given places
- Do NOT invent places
- Choose places that best fit the trip duration
- Output ONLY a JSON array of place_ids in visiting order
- No explanations
"""
            },
            {
                "role": "user",
                "content": f"""
City: {city}
Days: {days}

Available places:
{json.dumps(place_summaries, indent=2)}

User request:
{question}

Return JSON like:
["PLACE_ID_1", "PLACE_ID_2", "..."]
"""
            }
        ],
        options={"temperature": 0.3}
    )

    return json.loads(response["message"]["content"])

# ===============================
# 🧠 Main Planner
# ===============================

def plan_trip(city: str, question: str) -> TripPlan:
    title = f"Trip Plan for {city.title()}"
    description = f"A personalized trip plan for {city.title()} based on your preferences."

    places = fetch_places_by_city(city)
    if not places:
        raise ValueError("No places found")

    days = extract_days(question)
    people = extract_people(question)

    # LLM decides order + selection
    selected_place_ids = ask_llm_to_plan(
        places=places,
        days=days,
        city=city,
        question=question
    )

    # Preserve original place objects
    place_map = {p["place_id"]: p for p in places}

    top_places = [
        place_map[pid]
        for pid in selected_place_ids
        if pid in place_map
    ]

    day_groups = split_into_days(top_places)

    tourist_spots = [
        TouristSpot(
            name=p["name"],
            popularity=p.get("popularity", "NOT_AVAILABLE"),
            description=p.get("description", "NOT_AVAILABLE")
        )
        for p in top_places
    ]

    transport_entries = []
    total_cost = 0

    for day_places in day_groups:
        for i in range(len(day_places) - 1):
            routes = get_transport(
                day_places[i]["place_id"],
                day_places[i + 1]["place_id"]
            )
            if not routes:
                continue

            best = routes[0]
            transport_entries.append(
                Transport(
                    name=best["mode"],
                    average_cost=best["cost"],
                    duration=best["duration"]
                )
            )

            total_cost += best["cost"]


    instructions = build_instruction_steps(day_groups)

    return TripPlan(
        title=title,
        description=description,
        people=people,
        tourist_spots=tourist_spots,
        transport=transport_entries,
        instructions=instructions,
        cost_summary=CostSummary(
            total_cost_for_people=total_cost * people,
            people_count=people
        )
    )
