from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:
    END = None
    StateGraph = None

from agents import narration_agent, planner_agent, transport_agent


class TripPlan(BaseModel):
    title: str
    city: str
    days: int
    people: int
    budget: Optional[float] = None
    plans: List[Dict[str, Any]]
    error_message: Optional[str] = None


class OrchestrationState(TypedDict, total=False):
    city: str
    days: int
    people: int
    budget: Optional[float]
    cost_budget: Optional[float]
    coordinates: Optional[Dict[str, float]]
    excluded_places: List[str]
    places: List[Dict[str, Any]]
    estimated_cost: float
    transportation_route: List[Any]
    transportation_cost: float
    plans: List[Dict[str, Any]]
    force_economy: bool
    iterations: int
    max_iterations: int
    error_message: str


def get_place_visit_cost(place):
    if isinstance(place, dict):
        name = place.get("name", "").lower()
        cost = place.get("avg_cost_level", 0)
    else:
        name = str(place).lower()
        cost = 0

    if any(key in name for key in ["beach", "park", "garden", "temple"]):
        return 0.0

    return float(cost or 0)


def get_calculated_budget(state: OrchestrationState):
    plans = state.get("plans", []) or []
    if plans:
        return float(plans[0].get("cost_summary", {}).get("total_cost_for_people", 0) or 0)

    people = max(int(state.get("people", 1) or 1), 1)
    estimated_cost = float(state.get("estimated_cost", 0) or 0)
    transportation_cost = float(state.get("transportation_cost", 0) or 0)
    return round((estimated_cost + transportation_cost) * people, 2)


def remove_place_for_budget(state: OrchestrationState):
    places = state.get("places", []) or []
    if len(places) <= 1:
        return state

    removed_place = min(
        places,
        key=lambda place: (
            place.get("popularity", 0),
            -get_place_visit_cost(place),
            place.get("name", ""),
        ),
    )
    removed_name = removed_place.get("name")

    excluded_places = state.get("excluded_places", [])
    if removed_name and removed_name not in excluded_places:
        excluded_places.append(removed_name)

    state["excluded_places"] = excluded_places
    state["places"] = [place for place in places if place.get("name") != removed_name]
    state["estimated_cost"] = sum(get_place_visit_cost(place) for place in state["places"])
    print(f"Budget optimization removed place: {removed_name}")
    return state


def retrieve_budgeted_places(state: OrchestrationState):
    place_state = planner_agent.app.invoke({
        "city": state["city"],
        "days": state.get("days", 1),
        "cost_budget": state.get("budget") or state.get("cost_budget"),
        "excluded_places": state.get("excluded_places", []),
        "coordinates": state.get("coordinates"),
        "places": [],
        "estimated_cost": 0.0,
    })

    state["places"] = place_state.get("places", [])
    state["estimated_cost"] = place_state.get("estimated_cost", 0.0)

    if not state["places"]:
        state["error_message"] = "No places found within the provided budget."

    return state


def generate_transport(state: OrchestrationState):
    if not state.get("places"):
        state["transportation_route"] = []
        state["transportation_cost"] = 0.0
        return state

    transport_state = transport_agent.transport_app.invoke({
        "city": state.get("city"),
        "places": state.get("places", []),
        "estimated_cost": state.get("estimated_cost", 0.0),
        "cost_budget": state.get("budget") or state.get("cost_budget"),
        "transportation_route": [],
        "transportation_cost": 0.0,
        "force_economy": bool(state.get("force_economy", False)),
    })

    state["places"] = transport_state.get("places", state.get("places", []))
    state["estimated_cost"] = transport_state.get("estimated_cost", state.get("estimated_cost", 0.0))
    state["transportation_route"] = transport_state.get("transportation_route", [])
    state["transportation_cost"] = transport_state.get("transportation_cost", 0.0)

    if transport_state.get("error_message"):
        state["error_message"] = transport_state["error_message"]

    return state


def generate_narration(state: OrchestrationState):
    if not state.get("places"):
        state["plans"] = []
        return state

    narration_state = narration_agent.narration_app.invoke({
        "city": state.get("city"),
        "people": state.get("people", 1),
        "budget": state.get("budget"),
        "places": state.get("places", []),
        "transportation_route": state.get("transportation_route", []),
        "transportation_cost": state.get("transportation_cost", 0.0),
        "estimated_cost": state.get("estimated_cost", 0.0),
    })

    state["plans"] = narration_state.get("plans", [])

    if narration_state.get("error_message"):
        state["error_message"] = narration_state["error_message"]

    return state


def should_continue_after_places(state: OrchestrationState):
    return "success" if state.get("places") else "fail"


def budget_router(state: OrchestrationState):
    budget = state.get("budget") or state.get("cost_budget")
    if budget is None or budget <= 0:
        return "success"

    calculated_budget = get_calculated_budget(state)
    print("Calculated budget:", calculated_budget)
    print("User budget:", budget)

    if calculated_budget <= float(budget):
        return "success"

    iterations = int(state.get("iterations", 0) or 0)
    max_iterations = int(state.get("max_iterations", 6) or 6)

    if iterations >= max_iterations:
        return "fail"

    if not state.get("force_economy"):
        print("Budget exceeded. Retrying with economy transport.")
        return "retry_transport"

    print("Budget exceeded after economy transport. Trying different places.")
    return "retry_places"


def force_economy_transport(state: OrchestrationState):
    state["force_economy"] = True
    state["iterations"] = int(state.get("iterations", 0) or 0) + 1
    state["plans"] = []
    state["transportation_route"] = []
    state["transportation_cost"] = 0.0
    return state


def optimize_places_for_budget(state: OrchestrationState):
    state["iterations"] = int(state.get("iterations", 0) or 0) + 1
    remove_place_for_budget(state)
    state["plans"] = []
    state["transportation_route"] = []
    state["transportation_cost"] = 0.0
    state["force_economy"] = True
    return state


def mark_budget_failure(state: OrchestrationState):
    budget = state.get("budget") or state.get("cost_budget")
    calculated_budget = get_calculated_budget(state)
    state["error_message"] = (
        f"No itinerary fits the provided budget ({budget}). "
        f"Best calculated cost is {calculated_budget}."
    )
    return state


if StateGraph is not None:
    builder = StateGraph(OrchestrationState)
    builder.add_node("retrieve_budgeted_places", retrieve_budgeted_places)
    builder.add_node("generate_transport", generate_transport)
    builder.add_node("generate_narration", generate_narration)
    builder.add_node("force_economy_transport", force_economy_transport)
    builder.add_node("optimize_places_for_budget", optimize_places_for_budget)
    builder.add_node("mark_budget_failure", mark_budget_failure)

    builder.set_entry_point("retrieve_budgeted_places")
    builder.add_conditional_edges(
        "retrieve_budgeted_places",
        should_continue_after_places,
        {"success": "generate_transport", "fail": END},
    )
    builder.add_edge("generate_transport", "generate_narration")
    builder.add_conditional_edges(
        "generate_narration",
        budget_router,
        {
            "success": END,
            "retry_transport": "force_economy_transport",
            "retry_places": "optimize_places_for_budget",
            "fail": "mark_budget_failure",
        },
    )
    builder.add_edge("force_economy_transport", "generate_transport")
    builder.add_edge("optimize_places_for_budget", "retrieve_budgeted_places")
    builder.add_edge("mark_budget_failure", END)

    orchestration_graph = builder.compile()
else:
    class OrchestrationGraph:
        def invoke(self, state):
            state = retrieve_budgeted_places(state)
            if state.get("places"):
                state = generate_transport(state)
                state = generate_narration(state)
                route = budget_router(state)
                while route not in {"success", "fail"}:
                    if route == "retry_transport":
                        state = force_economy_transport(state)
                        state = generate_transport(state)
                    elif route == "retry_places":
                        state = optimize_places_for_budget(state)
                        state = retrieve_budgeted_places(state)
                        state = generate_transport(state)
                    state = generate_narration(state)
                    route = budget_router(state)
                if route == "fail":
                    state = mark_budget_failure(state)
            return state

    orchestration_graph = OrchestrationGraph()


def plan_trip(
    cityName: str,
    numberOfDays: int = 4,
    budget: Optional[float] = None,
    people: int = 4,
    coordinates: Optional[dict] = None,
) -> Dict[str, Any]:
    if not cityName or not isinstance(cityName, str) or not cityName.strip():
        raise ValueError("cityName is required")

    days = max(int(numberOfDays or 1), 1)
    people_count = max(int(people or 1), 1)

    result = orchestration_graph.invoke({
        "city": cityName.strip(),
        "days": days,
        "people": people_count,
        "budget": budget,
        "cost_budget": budget,
        "coordinates": coordinates,
        "excluded_places": [],
        "places": [],
        "plans": [],
        "force_economy": False,
        "iterations": 0,
        "max_iterations": max(days * 4, 6),
    })

    return {
        "title": f"Trip to {cityName.title()} for {days} Days",
        "city": cityName,
        "days": days,
        "people": people_count,
        "budget": budget,
        "plans": result.get("plans", []),
        "error_message": result.get("error_message"),
    }


if __name__ == "__main__":
    print(plan_trip("Chennai", numberOfDays=1, budget=4500, people=2))
