from typing import TypedDict, Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

from langchain_community.embeddings import OllamaEmbeddings

# ---------------- QDRANT SETUP ----------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)

client = QdrantClient(
    url="http://qdrant:6333"
)

# ---------------- STATE ----------------

class PlannerState(TypedDict):
    city: str
    days: int
    places: List[Dict[str, Any]]
    excluded_places: List[str]
    cost_budget: float
    estimated_cost: float
    coordinates: Optional[Dict[str, float]]

# ---------------- HELPERS ----------------

ACCESSIBILITY_ALIASES = {
    "Car": "Cab",
    "Auto": "Cab",
    "Cab": "Cab",
    "Bus": "Bus",
    "Train": "Train",
    "Walk": "Walk",
}


def normalize_accessibility(modes):
    if not isinstance(modes, list):
        return []

    normalized = []
    for mode in modes:
        if not isinstance(mode, str):
            continue

        normalized_mode = mode.strip().title()
        if normalized_mode in ACCESSIBILITY_ALIASES:
            normalized.append(ACCESSIBILITY_ALIASES[normalized_mode])

    return list(set(normalized))

def get_place_visit_cost(place):

    if isinstance(place, dict):

        name = place.get("name", "").lower()

        cost = place.get("avg_cost_level", 0)

    else:

        name = str(place).lower()

        cost = 0

    # Free places
    if any(
        k in name
        for k in ["beach", "park", "garden", "temple"]
    ):
        return 0.0

    return float(cost or 0)

# ---------------- NODE 1 ----------------

def retrieve_places(state: PlannerState):

    city = state["city"].strip().lower()

    days = state.get("days", 1)

    excluded_places = state.get(
        "excluded_places",
        []
    )

    k = min(days * 5, 20)

    city_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.city",
                match=MatchValue(value=city)
            )
        ]
    )

    print("\nRetrieving places from Qdrant...")

    results = client.query_points(
        collection_name="tourism_places",

        query=embeddings.embed_query(
            f"best tourist places in {city}"
        ),

        query_filter=city_filter,

        limit=k,

        with_payload=True
    )

    places = []

    for p in results.points:

        payload = getattr(p, "payload", {})

        metadata = payload.get("metadata", {})

        place_name = (
            metadata.get("place")
            or payload.get("place")
            or "Unknown Place"
        )

        # Skip excluded places
        if place_name in excluded_places:
            continue

        place_data = {
            "name": place_name,

            "avg_cost_level": metadata.get(
                "avg_cost_level",
                0
            ),

            "popularity": metadata.get(
                "popularity",
                5
            ),

            "description": payload.get(
                "page_content",
                "A tourist attraction."
            ),

            "location": metadata.get(
                "location"
            ),

            "accessibility": normalize_accessibility(
                metadata.get("accessibility", [])
            ),

            "area": metadata.get("area") or metadata.get("area_name"),

            "locality": metadata.get("locality"),

            "vicinity": metadata.get("vicinity"),

            "address": metadata.get("address")
        }

        places.append(place_data)

    # Sort by popularity descending
    places = sorted(
        places,
        key=lambda x: x.get("popularity", 0),
        reverse=True
    )

    state["places"] = places

    print("\nRetrieved Places:")

    for p in places:
        print(
            f"{p['name']} | "
            f"Popularity={p['popularity']} | "
            f"Cost={p['avg_cost_level']}"
        )

    return state

# ---------------- NODE 2 ----------------

def evaluate_budget(state: PlannerState):

    places = state.get("places", [])

    total_cost = sum(
        get_place_visit_cost(p)
        for p in places
    )

    state["estimated_cost"] = total_cost

    print("\nEstimated Cost:", total_cost)

    print(
        "Budget:",
        state["cost_budget"]
    )

    return state

# ---------------- ROUTER ----------------

def budget_router(state: PlannerState):

    total_cost = state["estimated_cost"]

    budget = state.get("cost_budget")

    if budget is None or budget <= 0:
        print("\nNo budget limit provided.")
        return "success"

    if total_cost > budget:

        print(
            "\nOver budget. "
            "Starting optimization..."
        )

        return "optimize"

    print("\nWithin budget.")

    return "success"

# ---------------- NODE 3 ----------------

def optimize_places(state: PlannerState):

    places = state.get("places", [])

    if not places:
        return state

    # Remove least popular expensive place

    sorted_places = sorted(
        places,
        key=lambda x: (
            x.get("popularity", 0),
            -x.get("avg_cost_level", 0)
        )
    )

    removed_place = sorted_places[0]

    print(
        "\nRemoving place:",
        removed_place["name"]
    )

    remaining_places = [
        p for p in places
        if p["name"] != removed_place["name"]
    ]

    excluded = state.get(
        "excluded_places",
        []
    )

    excluded.append(
        removed_place["name"]
    )

    state["excluded_places"] = excluded

    state["places"] = remaining_places

    return state

# ---------------- BUILD GRAPH ----------------

graph = StateGraph(PlannerState)

graph.add_node(
    "retrieve_places",
    retrieve_places
)

graph.add_node(
    "evaluate_budget",
    evaluate_budget
)

graph.add_node(
    "optimize_places",
    optimize_places
)

# Entry point
graph.set_entry_point(
    "retrieve_places"
)

# Main flow
graph.add_edge(
    "retrieve_places",
    "evaluate_budget"
)

# Conditional routing
graph.add_conditional_edges(
    "evaluate_budget",
    budget_router,
    {
        "success": END,
        "optimize": "optimize_places"
    }
)

# Optimization loop
graph.add_edge(
    "optimize_places",
    "evaluate_budget"
)

# ---------------- COMPILE ----------------

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "city": "Chennai",
        "days": 1,
        "cost_budget": 4500.0,
        "excluded_places": [],
        "coordinates": None
    })

    print("\n================ FINAL ITINERARY ================\n")

    for p in result["places"]:
        print(
            f"{p['name']} "
            f"(Cost={p['avg_cost_level']}, "
            f"Popularity={p['popularity']})"
        )

    print(
        "\nFinal Estimated Cost:",
        result["estimated_cost"]
    )
