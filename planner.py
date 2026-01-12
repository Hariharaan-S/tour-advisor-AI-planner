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
    tourist_spots: List[TouristSpot]
    transport: List[Transport]
    instructions: List[InstructionStep]
    cost_summary: CostSummary


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

STRICT RULES:
- You ONLY describe travel steps using the provided route facts
- You MUST write plain text instructions (NOT JSON, NOT code)
- Each instruction must be one clear sentence
- Do NOT invent places, transport modes, costs, or durations
- Do NOT add time slots unless explicitly present in the facts
- Do NOT repeat places unnecessarily
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
Write step-by-step travel instructions.
Each instruction MUST be on a new line.
Use ONLY the route facts above.
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
    places = fetch_places_by_city(city)
    if not places:
        raise ValueError("No places found")

    k = extract_k(question)
    people = extract_people(question)

    top_places = select_top_k_places(places, k)

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

    for i in range(len(top_places) - 1):
        src = top_places[i]
        dest = top_places[i + 1]

        routes = get_transport(src["place_id"], dest["place_id"])
        if not routes:
            continue

        best = routes[0]
        transport_entries.append(Transport(**best))
        total_cost += best["average_cost"]

        route_facts.append({
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
        tourist_spots=tourist_spots,
        transport=transport_entries,
        instructions=instructions,
        cost_summary=CostSummary(
            total_cost_for_people=total_cost * people,
            people_count=people
        )
    )
