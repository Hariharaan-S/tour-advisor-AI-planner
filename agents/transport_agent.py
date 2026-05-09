import hashlib
import json
import os
import re
from urllib.parse import quote_plus
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
import redis

load_dotenv()

try:
    from google_maps_tool import get_distance_matrix
except ModuleNotFoundError as import_error:
    google_maps_import_error = str(import_error)

    def get_distance_matrix(places, city=None):
        raise RuntimeError(f"Google Maps tool is unavailable: {google_maps_import_error}")

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:
    END = None
    StateGraph = None

try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    MongoClient = None


r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)


class Route:
    def __init__(self, origin, destination, distance_km, duration_min, mode="Bus", cost=0):
        self.origin = origin
        self.destination = destination
        self.distance_km = round(float(distance_km or 0), 2)
        self.duration_min = round(float(duration_min or 0), 2)
        self.mode = mode
        self.cost = round(float(cost or 0), 2)
        self.bus_route_id = None

    def to_dict(self):
        return {
            "origin": self.origin,
            "destination": self.destination,
            "distance_km": self.distance_km,
            "duration_min": self.duration_min,
            "mode": self.mode,
            "cost": self.cost,
            "bus_route_id": self.bus_route_id,
        }


class TransportState(TypedDict, total=False):
    city: str
    places: List[Dict[str, Any]]
    transportation_route: List[Route]
    transportation_cost: float
    estimated_cost: float
    cost_budget: float
    matrix: Dict[Any, Any]
    force_economy: bool
    mongo_uri: str
    bus_routes_collection: str
    error_message: str


TRAVEL_FARE_CHART = {
    "Cab": {"Low": 80, "Mid": 200, "High": 500},
    "Bus": {"Low": 10, "Mid": 45, "High": 80},
    "Train (MRTS)": {"Low": 5, "Mid": 10, "High": 20},
    "Train (Metro)": {"Low": 10, "Mid": 30, "High": 50},
    "Walk": {"Low": 0, "Mid": 0, "High": 0},
}


def generate_cache_key(places, city):
    place_names = [p["name"] if isinstance(p, dict) else p for p in places]
    key_string = f"{city}:{','.join(sorted(place_names))}"
    return "distance_matrix:" + hashlib.md5(key_string.encode()).hexdigest()


def serialize_matrix(data):
    return {f"{k[0]}||{k[1]}": v for k, v in data.items()}


def deserialize_matrix(data):
    return {tuple(k.split("||")): v for k, v in data.items()}


def normalize_name(value):
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    return value or None


def get_place_visit_cost(place):
    name = place.get("name", "").lower() if isinstance(place, dict) else str(place).lower()
    cost = place.get("avg_cost_level", 0) if isinstance(place, dict) else 0

    if any(k in name for k in ["beach", "park", "garden", "temple"]):
        return 0.0

    return float(cost or 0)


def get_estimated_cost(state):
    if state.get("estimated_cost") is not None:
        return float(state.get("estimated_cost") or 0)
    return sum(get_place_visit_cost(place) for place in state.get("places", []) or [])


def get_travel_cost(mode, distance_km):
    band = "Low" if distance_km <= 3 else "Mid" if distance_km <= 15 else "High"

    if mode == "Train":
        variant = "Train (Metro)" if distance_km <= 10 else "Train (MRTS)"
        return TRAVEL_FARE_CHART[variant][band], variant

    return TRAVEL_FARE_CHART.get(mode, TRAVEL_FARE_CHART["Cab"])[band], mode


def get_leg_matrix_entry(matrix, origin_name, destination_name, city=None):
    origin_name = normalize_name(origin_name)
    destination_name = normalize_name(destination_name)

    if not matrix or not origin_name or not destination_name:
        return None

    origin_variants = [origin_name]
    destination_variants = [destination_name]

    if city:
        city = city.strip()
        if city and city.lower() not in origin_name.lower():
            origin_variants.append(f"{origin_name}, {city}")
        if city and city.lower() not in destination_name.lower():
            destination_variants.append(f"{destination_name}, {city}")

    for origin in origin_variants:
        for destination in destination_variants:
            entry = matrix.get((origin, destination))
            if entry is not None:
                return entry

    return None


def choose_travel_mode(origin, destination, distance_km, force_economy=False):
    origin_modes = set(origin.get("accessibility", []))
    destination_modes = set(destination.get("accessibility", []))
    shared_modes = origin_modes.intersection(destination_modes)
    priority = ("Train", "Bus", "Cab", "Walk") if force_economy else ("Bus", "Train", "Cab", "Walk")
    mode = next((candidate for candidate in priority if candidate in shared_modes), "Cab")
    cost, display_name = get_travel_cost(mode, distance_km)
    return display_name, cost


def get_place_lookup(places):
    lookup = {}
    for place in places or []:
        name = normalize_name(place.get("name"))
        if name:
            lookup[name.lower()] = place
    return lookup


def get_stop_name_candidates(place, fallback_name):
    candidates = []

    def add(value):
        value = normalize_name(value)
        if value and value.lower() not in {candidate.lower() for candidate in candidates}:
            candidates.append(value)

    add(fallback_name)

    if isinstance(place, dict):
        add(place.get("name"))
        for key in ("area", "area_name", "locality", "neighborhood", "vicinity", "address"):
            add(place.get(key))

        metadata = place.get("metadata")
        if isinstance(metadata, dict):
            for key in ("area", "area_name", "locality", "neighborhood", "vicinity", "address"):
                add(metadata.get(key))

    fallback_parts = normalize_name(fallback_name)
    if fallback_parts:
        words = [word for word in re.split(r"[^A-Za-z0-9]+", fallback_parts) if len(word) > 2]
        for word in words:
            if word.lower() not in {"chennai", "temple", "beach", "park", "museum"}:
                add(word)

    return candidates


def find_bus_route_id(routes_collection, origin_candidates, destination_candidates):
    for origin in origin_candidates:
        for destination in destination_candidates:
            query = {
                "$or": [
                    {
                        "source": {"$regex": re.escape(origin), "$options": "i"},
                        "destination": {"$regex": re.escape(destination), "$options": "i"},
                    },
                    {
                        "source": {"$regex": re.escape(destination), "$options": "i"},
                        "destination": {"$regex": re.escape(origin), "$options": "i"},
                    },
                ]
            }

            print("Bus route Mongo query:", query)
            document = routes_collection.find_one(query)

            if document and document.get("route_id") is not None:
                return document["route_id"]

    return None


def compute_matrix(state: TransportState):
    places = state.get("places", []) or []

    if len(places) < 2:
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
            matrix = get_distance_matrix(places, city)
            r.set(cache_key, json.dumps(serialize_matrix(matrix)), ex=86400)
            state["matrix"] = matrix

        print("Distance matrix size:", len(state["matrix"]))

    except Exception as e:
        print(f"Error fetching distance matrix: {e}")
        state["matrix"] = {}

    return state


def build_transport_routes(places, matrix, city=None, force_economy=False):
    routes = []
    total_cost = 0.0

    for index in range(len(places) - 1):
        origin = places[index]
        destination = places[index + 1]
        leg = get_leg_matrix_entry(matrix, origin.get("name"), destination.get("name"), city=city)

        if leg is None:
            print(f"Missing distance matrix entry: {origin.get('name')} -> {destination.get('name')}")
            continue

        distance_km = leg.get("distance_km", 0)
        duration_min = leg.get("duration_min", 0)
        mode, cost = choose_travel_mode(origin, destination, distance_km, force_economy=force_economy)

        route = Route(
            origin=origin.get("name"),
            destination=destination.get("name"),
            distance_km=distance_km,
            duration_min=duration_min,
            mode=mode,
            cost=cost,
        )
        routes.append(route)
        total_cost += route.cost

    return routes, round(total_cost, 2)


def remove_lowest_value_place(places):
    if len(places) <= 1:
        return []

    removed = min(
        places,
        key=lambda place: (
            place.get("popularity", 0),
            -get_place_visit_cost(place),
            place.get("name", ""),
        ),
    )
    print("Removing place to fit transport budget:", removed.get("name"))
    return [place for place in places if place.get("name") != removed.get("name")]


def generate_transportation_route(state: TransportState):
    places = state.get("places", []) or []
    matrix = state.get("matrix", {}) or {}
    cost_budget = state.get("cost_budget")
    force_economy = bool(state.get("force_economy", False))

    while len(places) >= 2:
        routes, transportation_cost = build_transport_routes(
            places,
            matrix,
            city=state.get("city"),
            force_economy=force_economy,
        )

        if not routes:
            state["places"] = places
            state["transportation_route"] = []
            state["transportation_cost"] = 0.0
            state["estimated_cost"] = sum(get_place_visit_cost(place) for place in places)
            state["error_message"] = "No transportation routes generated from the Google distance matrix."
            print(state["error_message"])
            return state

        estimated_cost = sum(get_place_visit_cost(place) for place in places)
        total_cost = estimated_cost + transportation_cost

        if cost_budget is None or total_cost <= float(cost_budget):
            state["places"] = places
            state["transportation_route"] = routes
            state["transportation_cost"] = transportation_cost
            state["estimated_cost"] = round(estimated_cost, 2)
            print("Transportation cost:", transportation_cost)
            print("Estimated + transportation cost:", round(total_cost, 2))
            return state

        if not force_economy:
            print("Over budget. Retrying with economy transport.")
            force_economy = True
            state["force_economy"] = True
            continue

        places = remove_lowest_value_place(places)

    state["places"] = places
    state["transportation_route"] = []
    state["transportation_cost"] = 0.0
    state["estimated_cost"] = sum(get_place_visit_cost(place) for place in places)
    state["error_message"] = f"No transportation route fits the provided budget ({cost_budget})."
    print(state["error_message"])
    return state


def assign_bus_route_ids(state: TransportState):
    routes = state.get("transportation_route", []) or []
    bus_routes = [route for route in routes if (normalize_name(route.mode) or "").lower() == "bus"]

    if not bus_routes:
        return state

    if MongoClient is None:
        print("Skipping bus route lookup: pymongo is not installed.")
        return state

    mongo_user = os.getenv("MONGODBUSER")
    mongo_password = quote_plus(os.getenv("MONGODBPASSWORD"))

    mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@mongodb:27017/tour_advisor?authSource=admin"

    try:
        client = MongoClient(
            state.get("mongo_uri", mongo_uri),
            serverSelectionTimeoutMS=2000
        )
        collection_name = state.get("bus_routes_collection", "new_route_details")
        collection = client["bus_routes"][collection_name]
        print(f"Using Mongo collection: bus_routes.{collection_name}")
        place_lookup = get_place_lookup(state.get("places", []))

        for route in bus_routes:
            origin_place = place_lookup.get(str(route.origin).lower())
            destination_place = place_lookup.get(str(route.destination).lower())
            origin_candidates = get_stop_name_candidates(origin_place, route.origin)
            destination_candidates = get_stop_name_candidates(destination_place, route.destination)

            route.bus_route_id = find_bus_route_id(
                collection,
                origin_candidates,
                destination_candidates,
            )

            if route.bus_route_id is None:
                print(
                    "No bus route match for "
                    f"{route.origin} -> {route.destination}. "
                    f"Tried origins={origin_candidates}, destinations={destination_candidates}"
                )
            else:
                print(f"Bus route matched: {route.origin} -> {route.destination} = {route.bus_route_id}")

        client.close()

    except Exception as e:
        print(f"Error fetching bus route IDs: {e}")

    return state


def budget_router(state: TransportState):
    cost_budget = state.get("cost_budget")
    if cost_budget is None:
        return "success"

    total_cost = get_estimated_cost(state) + float(state.get("transportation_cost", 0) or 0)
    return "success" if total_cost <= float(cost_budget) else "fail"


if StateGraph is not None:
    builder = StateGraph(TransportState)
    builder.add_node("compute_matrix", compute_matrix)
    builder.add_node("generate_transportation_route", generate_transportation_route)
    builder.add_node("assign_bus_route_ids", assign_bus_route_ids)

    builder.set_entry_point("compute_matrix")
    builder.add_edge("compute_matrix", "generate_transportation_route")
    builder.add_edge("generate_transportation_route", "assign_bus_route_ids")
    builder.add_conditional_edges("assign_bus_route_ids", budget_router, {"success": END, "fail": END})

    transport_app = builder.compile()
else:
    class TransportApp:
        def invoke(self, state):
            state = compute_matrix(state)
            state = generate_transportation_route(state)
            return assign_bus_route_ids(state)

    transport_app = TransportApp()


def plan_transport(
    places: List[Dict[str, Any]],
    city: Optional[str] = None,
    estimated_cost: float = 0.0,
    cost_budget: Optional[float] = None,
) -> List[Route]:
    result = transport_app.invoke({
        "city": city,
        "places": places,
        "estimated_cost": estimated_cost,
        "cost_budget": cost_budget,
        "transportation_route": [],
        "transportation_cost": 0.0,
        "force_economy": False,
    })

    return result.get("transportation_route", [])


if __name__ == "__main__":
    sample_places = [
        {"name": "Marundeeswara Temple", "avg_cost_level": 0, "popularity": 9, "accessibility": ["Bus", "Cab"], "area": "Thiruvanmiyur"},
        {"name": "Chennai Central", "avg_cost_level": 10, "popularity": 7, "accessibility": ["Bus", "Cab"], "area": "Central"},
    ]

    print("Running transport agent sample...\n")
    planned_routes = plan_transport(sample_places, city="Chennai", cost_budget=500)

    print("\n================ TRANSPORT ROUTES ================\n")
    if not planned_routes:
        print("No transport routes generated.")
    for route in planned_routes:
        print(route.to_dict())
