# ---------- IMPORTS ----------
import json
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

def normalize_accessibility(modes):
    if not isinstance(modes, list): return []
    normalized = []
    for mode in modes:
        if not isinstance(mode, str): continue
        m = mode.strip().title()
        if m in ACCESSIBILITY_ALIASES: normalized.append(ACCESSIBILITY_ALIASES[m])
    return list(set(normalized))

def get_travel_cost(mode, distance_km):
    band = "Low" if distance_km <= 5 else "Mid" if distance_km <= 15 else "High"
    if mode == "Train":
        variant = "Train (Metro)" if distance_km <= 10 else "Train (MRTS)"
        return TRAVEL_FARE_CHART[variant][band], variant
    return TRAVEL_FARE_CHART.get(mode, TRAVEL_FARE_CHART["Cab"])[band], mode

def get_place_visit_cost(place):
    return float(place.get("avg_cost_level", 0) or 0)

def get_route_place_cost(route):
    return round(sum(get_place_visit_cost(place) for place in route), 2)

def get_leg_matrix_entry(matrix, origin_name, destination_name):
    return matrix.get((origin_name, destination_name))

def choose_travel_mode(origin, destination, distance_km, duration_min, force_economy=False):
    origin_modes = set(origin.get("accessibility", []))
    dest_modes = set(destination.get("accessibility", []))
    shared = origin_modes.intersection(dest_modes)
    priority = ("Train", "Bus", "Walk", "Cab") if force_economy else ("Cab", "Train", "Bus", "Walk")
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
            "accessibility": normalize_accessibility(metadata.get("accessibility", []))
        })

    state["places"] = places
    return state

def compute_matrix(state: PlannerState):
    places = state.get("places", []) or []
    if not places:
        state["matrix"] = {}
        return state

    try:
        state["matrix"] = get_distance_matrix(places, state.get("city"))
    except Exception as e:
        state["matrix"] = {}

    return state

def generate_routes(state: PlannerState):
    """QWEN LLM: Sequences the spots into intelligent versions."""
    places = state.get("places", []) or []
    # Fallback pure deterministic path if no places or LLM fails.
    if not places:
        state["routes"] = []
        return state

    names = [p.get("name", "") for p in places]
    prompt = f"As a travel expert for {state.get('city','unknown')}, create 2 routes from these spots: {names}. Version 1: Famous, Version 2: Efficient. Return ONLY JSON: {{\"version1\": [names], \"version2\": [names]}}"

    place_lookup = {p.get("name", ""): p for p in places}
    routes = []

    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)

        if isinstance(data, dict):
            for k in ["version1", "version2"]:
                if k in data and isinstance(data[k], list):
                    route_items = [place_lookup[n] for n in data[k] if n in place_lookup]
                    if route_items:
                        routes.append({"route": route_items})

    except Exception as e:
        print("generate_routes: LLM or parsing error:", e)

    if not routes:
        # fallback ranking: popularity descending and simple reverse
        forward = sorted(places, key=lambda p: p.get("popularity", 0), reverse=True)
        reverse = list(reversed(forward))
        routes.append({"route": forward})
        if reverse != forward:
            routes.append({"route": reverse})

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

        tourist_spots, transport = [], []
        total_travel_cost = 0
        total_place_cost = get_route_place_cost(route)

        for i, place in enumerate(route):
            tourist_spots.append({
                "name": place.get("name", ""),
                "popularity": place.get("popularity", 0),
                "description": place.get("description", "")
            })
            if i < len(route) - 1:
                d = get_leg_matrix_entry(matrix, place.get("name"), route[i+1].get("name")) or \
                    get_leg_matrix_entry(matrix, f"{place.get('name', '')}, {state.get('city','')} ", f"{route[i+1].get('name','')}, {state.get('city','')} ")
                if d:
                    leg = choose_travel_mode(place, route[i+1], d.get("distance_km", 0), d.get("duration_min", 0), force_economy=force_econ)
                    transport.append(leg)
                    total_travel_cost += leg.get("average_cost", 0)

        # QWEN GENERATES INSTRUCTIONS
        prompt = f"Generate a timed day-by-day itinerary for: {[p.get('name','') for p in route]}. Include Day, Time, and Description. Return ONLY JSON: {{\"instructions\": [{{\"day\": 1, \"time\": \"9:00 AM\", \"place_name\": \"...\", \"description\": \"...\"}}]}}"

        instructions = []
        try:
            llm_res = llm.invoke(prompt)
            instructions = json.loads(llm_res.content).get("instructions", [])
        except Exception as e:
            print("generate_itinerary: LLM instructions error:", e)
            instructions = [{"day": 1, "time": f"{9+i}:00 AM", "place_name": p.get("name", "")} for i, p in enumerate(route)]

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