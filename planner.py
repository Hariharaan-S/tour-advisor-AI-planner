# ---------- IMPORTS ----------
from datetime import datetime
import json
import math
import redis
import hashlib
from typing import TypedDict, Any, Dict, List, Optional

from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, 
    FormulaQuery, SumExpression, GaussDecayExpression, 
    DecayParamsExpression, GeoDistance, GeoDistanceParams, GeoPoint,
    Prefetch
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama

from google_maps_tool import get_distance_matrix
from datetime import datetime, timedelta


#------------------- Deterministic Time Engine -------------------------

def compute_itinerary_schedule(route, transport, start_time="09:00 AM"):
    """
    Deterministically computes arrival + departure times
    using opening hours from VISITING_TIMES.
    """

    def parse_time(t):
        if not isinstance(t, str):
            return None
        normalized = t.strip().replace(".", ":").upper()
        normalized = normalized.replace("NOON", "PM").replace("MIDNIGHT", "AM")
        if normalized == "0":
            return None
        if normalized in {"00:00", "00:00 AM", "00:00 PM"}:
            return datetime.strptime("12:00 AM", "%I:%M %p")
        try:
            return datetime.strptime(normalized, "%I:%M %p")
        except ValueError:
            try:
                return datetime.strptime(normalized, "%I:%M%p")
            except ValueError:
                return None

    def format_time(t):
        return t.strftime("%I:%M %p")

    def get_open_windows(place_name):
        if not isinstance(place_name, str):
            return []

        normalized = place_name.lower()
        place_key = None
        for candidate in VISITING_TIMES.keys():
            if candidate in normalized:
                place_key = candidate
                break

        if place_key is None:
            return []

        windows = []
        schedule = VISITING_TIMES.get(place_key, {})
        for period in ("morning", "evening"):
            period_data = schedule.get(period, {})
            open_time = _clean_time_label(period_data.get("open"))
            close_time = _clean_time_label(period_data.get("close"))
            start = parse_time(open_time)
            end = parse_time(close_time)
            if start and end:
                windows.append((start, end))

        return windows

    def align_to_open_window(current_time, windows):
        if not windows:
            return current_time

        day_base = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        actual_windows = []
        for start, end in windows:
            actual_start = day_base + timedelta(hours=start.hour, minutes=start.minute)
            actual_end = day_base + timedelta(hours=end.hour, minutes=end.minute)

            # If a window crosses midnight, advance its end into the next day.
            if actual_end <= actual_start:
                actual_end += timedelta(days=1)

            actual_windows.append((actual_start, actual_end))

        actual_windows.sort(key=lambda w: w[0])

        for start, end in actual_windows:
            if current_time <= start:
                return start
            if start <= current_time < end:
                return current_time

        # already past all windows today, move to next day's first window
        first_start = actual_windows[0][0] + timedelta(days=1)
        return first_start

    current_time = parse_time(start_time)
    if current_time is None:
        current_time = datetime.strptime("09:00 AM", "%I:%M %p")

    instructions = []

    for i, place in enumerate(route):
        place_name = place.get("name", "")

        if i > 0:
            leg = transport[i-1]
            travel_min = leg.get("duration", 0)
            current_time += timedelta(minutes=travel_min)

        arrival_time = current_time
        open_windows = get_open_windows(place_name)
        arrival_time = align_to_open_window(arrival_time, open_windows)

        visit_min = place.get("visit_duration_minutes", 60)
        departure_time = arrival_time + timedelta(minutes=visit_min)
        current_time = departure_time

        instructions.append({
            "day": arrival_time.day,
            "time": format_time(arrival_time),
            "place_name": place_name,
            "arrival_time": format_time(arrival_time),
            "departure_time": format_time(departure_time),
            "visit_duration": visit_min
        })

    return instructions


#-------------- Redis Client Connection --------------
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

#-------------- Redis Helper Functions --------------
def generate_cache_key(places, city):
    # handle both dict and string inputs
    place_names = [
        p["name"] if isinstance(p, dict) else p
        for p in places
    ]

    key_string = f"{city}:{','.join(sorted(place_names))}"
    return "distance_matrix:" + hashlib.md5(key_string.encode()).hexdigest()

def serialize_matrix(data):
    return {
        f"{k[0]}||{k[1]}": v   # convert tuple → string
        for k, v in data.items()
    }

def deserialize_matrix(data):
    return {
        tuple(k.split("||")): v
        for k, v in data.items()
    }

# ---------- LLM (QWEN) ----------
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0,
    format="json"
)

# ---------- QDRANT SETUP ----------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="tourism_places",
    embedding=embeddings
)

# ---------- MODELS ----------
class TripPlan(BaseModel):
    title: str               
    city: str
    days: int
    people: int
    budget: Optional[float] = None
    plans: List[Dict[str, Any]]

class PlannerState(TypedDict, total=False):
    city: str
    days: int
    people: int
    coordinates: dict
    budget: float
    places: list
    visiting_time: dict
    transport: list
    matrix: dict
    routes: list
    plans: list
    iterations: int          
    force_economy: bool      
    error_message: str

# ---------- CONSTANTS & HELPERS ----------
TRAVEL_FARE_CHART = {
    "Cab": {"Low": 80, "Mid": 200, "High": 500},
    "Bus": {"Low": 10, "Mid": 45, "High": 80},
    "Train (MRTS)": {"Low": 5, "Mid": 10, "High": 20},
    "Train (Metro)": {"Low": 10, "Mid": 30, "High": 50},
    "Walk": {"Low": 0, "Mid": 0, "High": 0},
}

ACCESSIBILITY_ALIASES = {
    "Car": "Cab", "Auto": "Cab", "Cab": "Cab",
    "Bus": "Bus", "Train": "Train", "Walk": "Walk",
}

#---------------- Average opening and closing time of places -----------------------------
VISITING_TIMES = {
    "beach": {
        "morning": {
            "open": "0",
            "close": "0"
        },
        "evening": {
            "open": "0",
            "close": "0"
        }
    },
    "temple": {
        "morning": {
            "open": "06:00 AM",
            "close": "12.00 NOON"
        },
        "evening": {
            "open":"04:30 PM",
            "close":"09.00 PM"
        }
    },
    "amusement park": {
        "morning": {
            "open": "09.30 AM",
            "close": "00:00"
        },
        "evening": {
            "open": "00:00",
            "close": "06:30 PM"
        }
    },
    "park": {
        "morning": {
            "open": "09.30 AM",
            "close": "00:00"
        },
        "evening": {
            "open": "00:00",
            "close": "06:30 PM"
        }
    }
}


def _clean_time_label(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value in {"0", "00.0", "00.00"}:
        return None
    return value


def get_visit_profile(place_name, visiting_time=VISITING_TIMES):
    place_key = None
    if isinstance(place_name, str):
        normalized = place_name.lower()
        for candidate in visiting_time.keys():
            if candidate in normalized:
                place_key = candidate
                break

    if place_key is None:
        return {
            "type": "general",
            "duration_min": 60,
            "opening_hours": ["Flexible timing"]
        }

    schedule = visiting_time.get(place_key, {})
    hours = []
    morning = schedule.get("morning", {})
    evening = schedule.get("evening", {})

    morning_open = _clean_time_label(morning.get("open"))
    morning_close = _clean_time_label(morning.get("close"))
    if morning_open and morning_close:
        hours.append(f"{morning_open} - {morning_close}")

    evening_open = _clean_time_label(evening.get("open"))
    evening_close = _clean_time_label(evening.get("close"))
    if evening_open and evening_close:
        hours.append(f"{evening_open} - {evening_close}")

    if not hours:
        hours = ["Flexible timing"]

    duration = 60
    if place_key == "temple":
        duration = 90
    elif place_key == "amusement park":
        duration = 240
    elif place_key == "park":
        duration = 120
    elif place_key == "beach":
        duration = 180

    return {
        "type": place_key,
        "duration_min": duration,
        "opening_hours": hours
    }


def normalize_accessibility(modes):
    if not isinstance(modes, list): return []
    normalized = []
    for mode in modes:
        if not isinstance(mode, str): continue
        m = mode.strip().title()
        if m in ACCESSIBILITY_ALIASES: normalized.append(ACCESSIBILITY_ALIASES[m])
    return list(set(normalized))

def get_travel_cost(mode, distance_km):
    band = "Low" if distance_km <= 3 else "Mid" if distance_km <= 15 else "High"
    if mode == "Train":
        variant = "Train (Metro)" if distance_km <= 10 else "Train (MRTS)"
        return TRAVEL_FARE_CHART[variant][band], variant
    return TRAVEL_FARE_CHART.get(mode, TRAVEL_FARE_CHART["Cab"])[band], mode

def get_place_visit_cost(place):
    if isinstance(place, dict):
        name = place.get("name", "").lower()
        cost = place.get("avg_cost_level", 0)
    else:
        name = str(place).lower()
        cost = 0

    if any(k in name for k in ["beach", "park", "garden", "temple"]):
        return 0.0

    return float(cost or 0)

def get_route_place_cost(route):
    return round(sum(get_place_visit_cost(place) for place in route), 2)

def get_leg_matrix_entry(matrix, origin_name, destination_name, city=None):
    if not matrix or not origin_name or not destination_name:
        return None

    origin_name = origin_name.strip() if isinstance(origin_name, str) else ""
    destination_name = destination_name.strip() if isinstance(destination_name, str) else ""
    if not origin_name or not destination_name:
        return None

    origin_variants = [origin_name]
    destination_variants = [destination_name]
    if city and isinstance(city, str):
        city = city.strip()
        if city and city.lower() not in origin_name.lower():
            origin_variants.append(f"{origin_name}, {city}")
        if city and city.lower() not in destination_name.lower():
            destination_variants.append(f"{destination_name}, {city}")

    for origin_variant in origin_variants:
        for destination_variant in destination_variants:
            entry = matrix.get((origin_variant, destination_variant))
            if entry is not None:
                return entry
    return None

def _is_valid_coordinate_pair(coords):
    return (
        isinstance(coords, dict)
        and isinstance(coords.get("lat"), (int, float))
        and isinstance(coords.get("lng"), (int, float))
    )

def _haversine_distance_km(origin, destination):
    """Returns great-circle distance between two lat/lng points."""
    lat1 = math.radians(origin["lat"])
    lon1 = math.radians(origin["lng"])
    lat2 = math.radians(destination["lat"])
    lon2 = math.radians(destination["lng"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * (2 * math.asin(math.sqrt(a)))

def _order_route_from_closest_start(route, coordinates):
    """Keeps route intent, but ensures the first stop is the nearest place to the user."""
    if not route or not _is_valid_coordinate_pair(coordinates):
        return route

    ranked = []
    for idx, place in enumerate(route):
        place_coords = place.get("location")
        if _is_valid_coordinate_pair(place_coords):
            ranked.append((_haversine_distance_km(coordinates, place_coords), idx, place))

    if not ranked:
        return route

    _, nearest_idx, _ = min(ranked, key=lambda item: item[:2])
    if nearest_idx == 0:
        return route

    return [route[nearest_idx], *route[:nearest_idx], *route[nearest_idx + 1:]]

def _sort_places_by_distance(places, coordinates):
    if not places:
        return []

    if not _is_valid_coordinate_pair(coordinates):
        return sorted(places, key=lambda p: p.get("popularity", 0), reverse=True)

    def sort_key(place):
        place_coords = place.get("location")
        if _is_valid_coordinate_pair(place_coords):
            return (_haversine_distance_km(coordinates, place_coords), -place.get("popularity", 0), place.get("name", ""))
        return (float("inf"), -place.get("popularity", 0), place.get("name", ""))

    return sorted(places, key=sort_key)

def _filter_places_by_budget(places, budget, people):
    """Filters places so cumulative visit cost stays within budget before routing."""
    if budget is None or budget <= 0 or not places:
        return places

    per_person_budget = float(budget) / max(int(people or 1), 1)
    running_total = 0.0
    filtered = []

    for place in sorted(places, key=lambda p: (get_place_visit_cost(p), -p.get("popularity", 0), p.get("name", ""))):
        place_cost = get_place_visit_cost(place)
        if filtered and running_total + place_cost > per_person_budget:
            continue
        if not filtered and place_cost > per_person_budget:
            continue
        filtered.append(place)
        running_total += place_cost

    return filtered

def choose_travel_mode(origin, destination, distance_km, duration_min, force_economy=False):
    origin_modes = set(origin.get("accessibility", []))
    dest_modes = set(destination.get("accessibility", []))
    shared = origin_modes.intersection(dest_modes)
    priority = ("Train", "Bus", "Cab", "Walk") if force_economy else ("Bus", "Train", "Cab", "Walk")
    chosen = next((m for m in priority if m in shared), "Cab")
    cost, display_name = get_travel_cost(chosen, distance_km)
    return {
        "name": display_name,
        "average_cost": cost,
        "duration": round(duration_min, 2),
        "distance_km": round(distance_km, 2),
        "origin": origin["name"],
        "destination": destination["name"]
    }

# ---------- NODES ----------

def retrieve_places(state):
    """QDRANT RAG: Fetches places based on City and optionally Geo-coordinates."""
    city = state["city"].strip().lower()
    k = min(state["days"] * 4, 15)
    coords = state.get("coordinates")
    city_filter = Filter(must=[FieldCondition(key="metadata.city", match=MatchValue(value=city))])

    if coords and "lat" in coords and "lng" in coords:
        results = client.query_points(
            collection_name="tourism_places",
            prefetch=[Prefetch(query=embeddings.embed_query(f"popular spots in {city}"), filter=city_filter, limit=40)],
            query=FormulaQuery(formula=SumExpression(sum=[GaussDecayExpression(
                gauss_decay=DecayParamsExpression(x=GeoDistance(geo_distance=GeoDistanceParams(
                    origin=GeoPoint(lat=coords["lat"], lon=coords["lng"]), to="metadata.location")), scale=5000))])),
            limit=k, with_payload=True
        )
    else:
        results = client.query_points(collection_name="tourism_places", query=embeddings.embed_query(f"tourist attractions in {city}"), 
                                      query_filter=city_filter, limit=k, with_payload=True)

    places = []
    for p in results.points:
        metadata = getattr(p, 'payload', {}).get('metadata', {}) if hasattr(p, 'payload') else {}
        name = metadata.get("place") or getattr(p, 'payload', {}).get('place') or "Unknown Place"
        places.append({
            "name": name,
            "avg_cost_level": metadata.get("avg_cost_level", 0),
            "popularity": metadata.get("popularity", 5),
            "description": getattr(p, 'payload', {}).get("page_content", "A famous attraction."),
            "accessibility": normalize_accessibility(metadata.get("accessibility", [])),
            "location": metadata.get("location")
        })

    state["places"] = places
    return state

def compute_matrix(state: PlannerState):
    places = state.get("places", []) or []
    if not places:
        state["matrix"] = {}
        return state

    try:
        city = state.get("city")
        cache_key = generate_cache_key(places, city)

        cached = r.get(cache_key)

        if cached:
            print("Cache HIT")
            state["matrix"] = deserialize_matrix(json.loads(cached))
        else:
            print("Cache MISS")
            data = get_distance_matrix(places, city)

            print("Distance matrix size:", len(data))
            print("Sample entry:", next(iter(data.items())) if data else "No data")

            # serialize before storing
            serializable_data = serialize_matrix(data)

            r.set(cache_key, json.dumps(serializable_data), ex=86400)

            state["matrix"] = data

    except Exception as e:
        print(f"Error fetching distance matrix: {e}")
        state["matrix"] = {}

    return state

def generate_routes(state: PlannerState):
    """Creates deterministic routes after budget filtering and distance sorting."""
    places = state.get("places", []) or []
    coordinates = state.get("coordinates", {}) or {}
    budget = state.get("budget")
    people = state.get("people", 1)

    if not places:
        state["routes"] = []
        return state

    filtered_places = _filter_places_by_budget(places, budget, people)
    if not filtered_places:
        state["routes"] = []
        return state

    primary_route = _sort_places_by_distance(filtered_places, coordinates)
    routes = [{"route": _order_route_from_closest_start(primary_route, coordinates)}]

    state["routes"] = routes
    return state

def generate_itinerary(state: PlannerState):
    """QWEN + LOGIC: Calculates costs and generates timed instructions."""
    routes = state.get("routes", []) or []
    people = state.get("people", 1)
    matrix = state.get("matrix", {})
    budget = state.get("budget")
    force_econ = state.get("force_economy", False)

    plans = []

    if not routes:
        state["plans"] = []
        return state

    for idx, r in enumerate(routes):
        route = r.get("route", []) or []
        if not route:
            continue

        tourist_spots = []
        city_name = str(state.get("city", "")).strip().lower()
        for p in route:
            name = str(p.get("name", "")).strip().lower()
            popularity = p.get("popularity", 0)
            tourist_spots.append({
                "name": name,
                "popularity": popularity,
                "description": f"place: {name} in {city_name}. popularity: {popularity}"
            })

        transport = []
        time_spent = []  # can be used for more advanced time calculations in the future
        total_travel_cost = 0
        total_place_cost = get_route_place_cost(route)

        for i, place in enumerate(route):

            if i < len(route) - 1:
                d = get_leg_matrix_entry(
                    matrix,
                    place.get("name"),
                    route[i+1].get("name"),
                    city=state.get("city")
                )

                if d is None:
                    origin_loc = place.get("location")
                    destination_loc = route[i+1].get("location")
                    if _is_valid_coordinate_pair(origin_loc) and _is_valid_coordinate_pair(destination_loc):
                        distance_km = _haversine_distance_km(origin_loc, destination_loc)
                        duration_min = max(10, distance_km / 40 * 60)
                        d = {"distance_km": round(distance_km, 2), "duration_min": round(duration_min, 2)}
                    else:
                        d = {"distance_km": 3.0, "duration_min": 18.0}

                leg = choose_travel_mode(
                    place,
                    route[i+1],
                    d.get("distance_km", 0),
                    d.get("duration_min", 0),
                    force_economy=force_econ
                )
                transport.append(leg)
                total_travel_cost += leg.get("average_cost", 0)

        route_with_duration = [
            {
                "name": p.get("name", ""),
                "visit_duration_minutes": p.get(
                    "visit_duration_minutes",
                    get_visit_profile(p.get("name", ""))["duration_min"]
                )
            }
            for p in route
        ]

        schedule = compute_itinerary_schedule(route_with_duration, transport)

        # QWEN GENERATES INSTRUCTIONS

        prompt = f"""
You are an expert trip planner and travel guide.

Your task is to generate an efficient, budget-aware itinerary that covers ALL the given places.

Places:
{[p.get('name','') for p in route]}
Transport Options:
{[f"{t.get('origin')} to {t.get('destination')} by {t.get('name')} (₹{t.get('average_cost', 0)}, {t.get('duration', 0)} mins)" for t in transport]}

Visit Details:
{time_spent}

Requirements:
- Plan the trip day-by-day with proper timing by using the visit durations and transport times given.
- Visit time should be allocated based on the type of place and its opening hours.
- Travel time between places should be calculated using the provided transport durations.
- Optimize route for minimum travel time and cost
- Include transport mode (bus/train/auto/walk etc.)
- Mention approximate travel cost for each step
- Ensure instructions are clear and sequential
- Cover ALL places without skipping
- Keep plan realistic (morning to evening schedule)

Output MUST include:
1. tourist_spots → list of place names
2. instructions → step-by-step travel plan

Instruction format example:
1. First take a bus to PLACE_1 which costs ₹X at HH:MM AM/PM
2. Spend time at PLACE_1, then take a TRANSPORT_MODE to PLACE_2 costing ₹Y at HH:MM AM/PM

Return ONLY valid JSON. No explanation.

Format:
{{
  "tourist_spots": [
    "PLACE_1",
    "PLACE_2"
  ],
  "instructions": [
    {{
      "day": 1,
      "time": "09:00 AM",
      "place_name": "PLACE_1",
      "description": "First take a bus to PLACE_1 which costs ₹X at 09:00 AM. Spend time there and then take another transport to next place."
    }}
  ]
}}
"""

        instructions = []
        try:
            llm_res = llm.invoke(prompt)
            instructions = json.loads(llm_res.content).get("instructions", [])
            if not isinstance(instructions, list):
                raise ValueError("Invalid instructions structure")
        except Exception as e:
            print("generate_itinerary: LLM instructions error:", e)
            instructions = []

        normalized_instructions = []

        for i, step in enumerate(schedule):
            leg = transport[i-1] if i > 0 else None

            description = ""

            if leg:
                description = (
                    f"Travel from {leg.get('origin')} to {leg.get('destination')} "
                    f"by {leg.get('name')} ({leg.get('duration')} mins, ₹{leg.get('average_cost', 0)}). "
                )

            description += (
                f"Arrive at {step['place_name']} at {step['arrival_time']}. "
                f"Spend {step['visit_duration']} minutes here. "
                f"Leave at {step['departure_time']}."
            )

            normalized_instructions.append({
                "day": step["day"],
                "time": step["arrival_time"],
                "place_name": step["place_name"],
                "description": description
            })
        instructions = normalized_instructions

        plans.append({
            "title": f"Trip to {state.get('city','').title()} - Route {idx+1}",
            "description": f"Optimized {'Economy' if force_econ else 'Standard'} itinerary",
            "tourist_spots": tourist_spots,
            "transport": transport,
            "instructions": instructions,
            "cost_summary": {"total_cost_for_people": round((total_place_cost + total_travel_cost) * people, 2), "people_count": people},
            "people": people
        })

    if budget is not None:
        plans = [p for p in plans if p.get("cost_summary", {}).get("total_cost_for_people", float('inf')) <= budget]

    state["plans"] = plans
    return state

# ---------- AGENT CONTROL ----------
def budget_evaluator(state: PlannerState):
    budget = state.get("budget")
    plans = state.get("plans", []) or []

    if not budget:
        return "success"

    valid_plans = [p for p in plans if p.get("cost_summary", {}).get("total_cost_for_people", float('inf')) <= budget]
    if valid_plans:
        state["plans"] = valid_plans
        return "success"

    if state.get("iterations", 0) < 1:
        return "optimize"

    state["error_message"] = f"No itinerary fits the provided budget ({budget})."
    return "fail"

def optimize_for_budget(state: PlannerState):
    state["force_economy"] = True
    state["iterations"] = state.get("iterations", 0) + 1
    state["places"] = sorted(state["places"], key=lambda x: x.get("avg_cost_level", 0))[:8]
    return state

# ---------- GRAPH ----------
builder = StateGraph(PlannerState)
builder.add_node("retrieve_places", retrieve_places)
builder.add_node("compute_matrix", compute_matrix)
builder.add_node("generate_routes", generate_routes)
builder.add_node("generate_itinerary", generate_itinerary)
builder.add_node("optimize_for_budget", optimize_for_budget)

builder.set_entry_point("retrieve_places")
builder.add_edge("retrieve_places", "compute_matrix")
builder.add_edge("compute_matrix", "generate_routes")
builder.add_edge("generate_routes", "generate_itinerary")
builder.add_conditional_edges("generate_itinerary", budget_evaluator, {"success": END, "optimize": "optimize_for_budget", "fail": END})
builder.add_edge("optimize_for_budget", "compute_matrix")

graph = builder.compile()

# ---------- FINAL WRAPPER ----------
def plan_trip(cityName: str, numberOfDays: int = 4, budget: Optional[float] = None, people: int = 4, coordinates: Optional[dict] = None) -> Dict[str, Any]:
    if not cityName or not isinstance(cityName, str) or not cityName.strip():
        raise ValueError("cityName is required")

    input_state = {
        "city": cityName.strip(),
        "days": max(int(numberOfDays), 1),
        "people": max(int(people), 1),
        "coordinates": coordinates,
        "budget": budget,
        "iterations": 0,
        "force_economy": False,
        "places": [],
        "routes": [],
        "matrix": {}
    }

    result = graph.invoke(input_state)

    return {
        "title": f"Trip to {cityName.title()} for {numberOfDays} Days",
        "city": cityName,
        "days": numberOfDays,
        "people": people,
        "budget": budget,
        "plans": result.get("plans", []),
        "error_message": result.get("error_message")
    }
