from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:
    END = None
    StateGraph = None


class NarrationState(TypedDict, total=False):
    city: str
    people: int
    budget: float
    places: List[Dict[str, Any]]
    transportation_route: List[Any]
    transportation_cost: float
    estimated_cost: float
    plans: List[Dict[str, Any]]
    error_message: str


VISITING_TIMES = {
    "beach": {
        "morning": {"open": "0", "close": "0"},
        "evening": {"open": "0", "close": "0"},
    },
    "temple": {
        "morning": {"open": "06:00 AM", "close": "12.00 NOON"},
        "evening": {"open": "04:30 PM", "close": "09.00 PM"},
    },
    "amusement park": {
        "morning": {"open": "09.30 AM", "close": "00:00"},
        "evening": {"open": "00:00", "close": "06:30 PM"},
    },
    "park": {
        "morning": {"open": "09.30 AM", "close": "00:00"},
        "evening": {"open": "00:00", "close": "06:30 PM"},
    },
}


def _clean_time_label(value):
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value or value in {"0", "00.0", "00.00"}:
        return None

    return value


def parse_time(value):
    if not isinstance(value, str):
        return None

    normalized = value.strip().replace(".", ":").upper()
    normalized = normalized.replace("NOON", "PM").replace("MIDNIGHT", "AM")

    if normalized == "0":
        return None

    if normalized in {"00:00", "00:00 AM", "00:00 PM"}:
        return datetime.strptime("12:00 AM", "%I:%M %p")

    for fmt in ("%I:%M %p", "%I:%M%p"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass

    return None


def format_time(value):
    return value.strftime("%I:%M %p")


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
            "opening_hours": ["Flexible timing"],
        }

    schedule = visiting_time.get(place_key, {})
    hours = []

    for period in ("morning", "evening"):
        period_data = schedule.get(period, {})
        open_time = _clean_time_label(period_data.get("open"))
        close_time = _clean_time_label(period_data.get("close"))
        if open_time and close_time:
            hours.append(f"{open_time} - {close_time}")

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
        "opening_hours": hours,
    }


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
        start = parse_time(_clean_time_label(period_data.get("open")))
        end = parse_time(_clean_time_label(period_data.get("close")))
        if start and end:
            windows.append((start, end))

    return windows


def align_to_open_window(current_time, windows, visit_min=0):
    if not windows:
        return current_time

    day_base = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    actual_windows = []

    for start, end in windows:
        actual_start = day_base + timedelta(hours=start.hour, minutes=start.minute)
        actual_end = day_base + timedelta(hours=end.hour, minutes=end.minute)

        if actual_end <= actual_start:
            actual_end += timedelta(days=1)

        actual_windows.append((actual_start, actual_end))

    actual_windows.sort(key=lambda window: window[0])
    visit_delta = timedelta(minutes=float(visit_min or 0))

    for start, end in actual_windows:
        if current_time <= start and start + visit_delta <= end:
            return start
        if start <= current_time < end and current_time + visit_delta <= end:
            return current_time

    return actual_windows[0][0] + timedelta(days=1)


def get_route_value(route, key, default=None):
    if isinstance(route, dict):
        return route.get(key, default)

    return getattr(route, key, default)


def normalize_transport(transportation_route):
    normalized = []

    for route in transportation_route or []:
        normalized.append({
            "name": get_route_value(route, "mode", "Bus"),
            "average_cost": float(get_route_value(route, "cost", 0) or 0),
            "duration": float(get_route_value(route, "duration_min", 0) or 0),
            "distance_km": float(get_route_value(route, "distance_km", 0) or 0),
            "origin": get_route_value(route, "origin", ""),
            "destination": get_route_value(route, "destination", ""),
            "bus_route_id": get_route_value(route, "bus_route_id"),
        })

    return normalized


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


def get_route_place_cost(route):
    return round(sum(get_place_visit_cost(place) for place in route), 2)


def compute_itinerary_schedule(route, transport, start_time="09:00 AM"):
    current_time = parse_time(start_time)
    if current_time is None:
        current_time = datetime.strptime("09:00 AM", "%I:%M %p")

    instructions = []

    for index, place in enumerate(route):
        place_name = place.get("name", "")

        if index > 0:
            leg = transport[index - 1] if index - 1 < len(transport) else {}
            current_time += timedelta(minutes=float(leg.get("duration", 0) or 0))

        visit_min = place.get("visit_duration_minutes")

        if visit_min is None:
            visit_min = get_visit_profile(place_name)["duration_min"]

        arrival_time = align_to_open_window(current_time, get_open_windows(place_name), visit_min=visit_min)
        departure_time = arrival_time + timedelta(minutes=float(visit_min or 60))
        current_time = departure_time

        instructions.append({
            "day": arrival_time.day,
            "time": format_time(arrival_time),
            "place_name": place_name,
            "arrival_time": format_time(arrival_time),
            "departure_time": format_time(departure_time),
            "visit_duration": int(visit_min or 60),
        })

    return instructions


def build_tourist_spots(places, city):
    city_name = str(city or "").strip().lower()
    tourist_spots = []

    for place in places or []:
        name = str(place.get("name", "")).strip().lower()
        popularity = place.get("popularity", 0)
        tourist_spots.append({
            "name": name,
            "popularity": popularity,
            "description": f"place: {name} in {city_name}. popularity: {popularity}",
        })

    return tourist_spots


def build_instructions(schedule, transport):
    instructions = []

    for index, step in enumerate(schedule):
        leg = transport[index - 1] if index > 0 and index - 1 < len(transport) else None
        description = ""

        if leg:
            bus_detail = ""
            if leg.get("name") == "Bus" and leg.get("bus_route_id"):
                bus_detail = f" route {leg.get('bus_route_id')}"

            description = (
                f"Travel from {leg.get('origin')} to {leg.get('destination')} "
                f"by {leg.get('name')}{bus_detail} with bus route id {leg.get('bus_route_id')} "
                f"({round(leg.get('duration', 0), 2)} mins, Rs.{round(leg.get('average_cost', 0), 2)}). "
            )

        description += (
            f"Arrive at {step['place_name']} at {step['arrival_time']}. "
            f"Spend {step['visit_duration']} minutes here. "
            f"Leave at {step['departure_time']}."
        )

        instructions.append({
            "day": step["day"],
            "time": step["arrival_time"],
            "place_name": step["place_name"],
            "description": description,
        })

    return instructions


def generate_narration(state: NarrationState):
    places = state.get("places", []) or []
    people = max(int(state.get("people", 1) or 1), 1)
    transport = normalize_transport(state.get("transportation_route", []) or [])

    if not places:
        state["plans"] = []
        state["error_message"] = "No places available for narration."
        return state

    route_with_duration = [
        {
            **place,
            "visit_duration_minutes": place.get(
                "visit_duration_minutes",
                get_visit_profile(place.get("name", ""))["duration_min"],
            ),
        }
        for place in places
    ]

    schedule = compute_itinerary_schedule(route_with_duration, transport)
    total_place_cost = get_route_place_cost(places)
    total_travel_cost = sum(leg.get("average_cost", 0) for leg in transport)

    state["plans"] = [{
        "title": f"Trip to {state.get('city', '').title()} - Route 1",
        "description": "Optimized Standard itinerary",
        "tourist_spots": build_tourist_spots(places, state.get("city", "")),
        "transport": transport,
        "instructions": build_instructions(schedule, transport),
        "cost_summary": {
            "total_cost_for_people": round((total_place_cost + total_travel_cost) * people, 2),
            "people_count": people,
        },
        "people": people,
    }]

    return state


if StateGraph is not None:
    builder = StateGraph(NarrationState)
    builder.add_node("generate_narration", generate_narration)
    builder.set_entry_point("generate_narration")
    builder.add_edge("generate_narration", END)
    narration_app = builder.compile()
else:
    class NarrationApp:
        def invoke(self, state):
            return generate_narration(state)

    narration_app = NarrationApp()


def narrate_trip(
    places: List[Dict[str, Any]],
    transportation_route: List[Any],
    city: Optional[str] = None,
    people: int = 1,
    budget: Optional[float] = None,
) -> Dict[str, Any]:
    return narration_app.invoke({
        "city": city,
        "people": people,
        "budget": budget,
        "places": places,
        "transportation_route": transportation_route,
    })


if __name__ == "__main__":
    sample_places = [
        {"name": "Marina Beach", "avg_cost_level": 0, "popularity": 9},
        {"name": "Kapaleeshwarar Temple", "avg_cost_level": 0, "popularity": 8},
        {"name": "Government Museum", "avg_cost_level": 10, "popularity": 7},
    ]
    sample_transport = [
        {
            "origin": "Marina Beach",
            "destination": "Kapaleeshwarar Temple",
            "distance_km": 3.5,
            "duration_min": 20,
            "mode": "Bus",
            "cost": 45,
            "bus_route_id": "1",
        },
        {
            "origin": "Kapaleeshwarar Temple",
            "destination": "Government Museum",
            "distance_km": 6.2,
            "duration_min": 35,
            "mode": "Cab",
            "cost": 200,
        },
    ]

    result = narrate_trip(sample_places, sample_transport, city="Chennai", people=2)
    for plan in result.get("plans", []):
        print(plan)
