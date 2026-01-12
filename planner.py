# planner.py
from typing import List, Union
from pydantic import BaseModel
import json
import ollama
import re

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
    step: int
    instruction: str

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
# 📂 Load Knowledge Base
# ===============================

with open("json_files/places.json", "r", encoding="utf-8") as f:
    PLACES = json.load(f)["data"]

with open("json_files/transport_distance.json", "r", encoding="utf-8") as f:
    TRANSPORT_DATA = json.load(f)["data"]


# ===============================
# 🛠 Helpers (Deterministic)
# ===============================

def extract_days(question: str, default=1):
    match = re.search(r"(\d+)\s*(day|days)", question.lower())
    if match:
        return int(match.group(1))
    return default


def fetch_places_by_city(city: str):
    city = city.strip().lower()
    return [
        p for p in PLACES
        if p.get("city", "").strip().lower() == city
    ]

def extract_k(question: str, default=3):
    for w in question.split():
        if w.isdigit():
            return int(w)
    return default

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
    routes = []
    for entry in TRANSPORT_DATA:
        if entry["src_place_id"] == src_id and entry["dest_place_id"] == dest_id:
            for stop in entry.get("nearby_stops", []):
                routes.append({
                    "name": stop["mode"],
                    "average_cost": stop["cost"],
                    "duration": stop.get("duration", "NOT_AVAILABLE")
                })
    return sorted(routes, key=lambda x: x["average_cost"])

def build_instruction_steps(text: str) -> List[InstructionStep]:
    steps = []
    for i, line in enumerate(text.split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        steps.append(
            InstructionStep(
                step=i,
                instruction=line
            )
        )
    return steps

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

DAY_START_MIN = 9.5 * 60   # 9:30 AM
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
# 🧹 JSON Sanitizer
# ===============================

def extract_json_only(text: str) -> str:
    # Find first JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON start found")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                raw = text[start:i + 1]
                break
    else:
        raise ValueError("No complete JSON found")

    # ---- Sanitize ----
    raw = raw.replace(": NOT_AVAILABLE", ': "NOT_AVAILABLE"')
    raw = raw.replace(": None", ': "NOT_AVAILABLE"')
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)

    return raw


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


def ask_llm_for_instructions(context_json, question):
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Route facts (DO NOT MODIFY):
{context_json}

User question:
{question}

TASK:
Narrate the trip DAY BY DAY using the route facts.
Start each day with "Day X:".
Then describe travel steps in order.
Each instruction must be one sentence on a new line.

"""
            }
        ],
        options={"temperature": 0.2}
    )
    return response["message"]["content"]


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

    MAX_SPOTS_PER_DAY = 4
    k = days * MAX_SPOTS_PER_DAY

    top_places = select_top_k_places(places, k)
    day_groups = split_into_days(top_places)



    tourist_spots: List[TouristSpot] = []
    transport_entries: List[Transport] = []
    total_cost = 0

    for p in top_places:
        tourist_spots.append(
            TouristSpot(
                name=p["name"],
                popularity=p.get("popularity", "NOT_AVAILABLE"),
                description=p.get("description", "NOT_AVAILABLE")
            )
        )

    route_facts = []

    for day_index, day_places in enumerate(day_groups, start=1):
        for i in range(len(day_places) - 1):
            src = day_places[i]
            dest = day_places[i + 1]

            routes = get_transport(src["place_id"], dest["place_id"])
            if not routes:
                continue

            best = routes[0]
            transport_entries.append(Transport(**best))
            total_cost += best["average_cost"]

            route_facts.append({
                "day": day_index,
                "from": src["name"],
                "to": dest["name"],
                "mode": best["name"],
                "cost": best["average_cost"],
                "duration": best["duration"]
            })


    # If no routes, skip LLM
    instructions = []

    if route_facts:
        context = {"route_facts": route_facts}
        llm_text = ask_llm_for_instructions(
            json.dumps(context, indent=2),
            question
        )
        instructions = build_instruction_steps(llm_text)


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
