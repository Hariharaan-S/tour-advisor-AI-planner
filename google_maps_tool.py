import googlemaps

with open('google_maps_api_key.txt','r') as f:
    api_key = f.readline()

gmaps = googlemaps.Client(key=api_key)

def get_place_coordinates(place_name, city=None):

    query = place_name

    if city and city.lower() not in query.lower():
        query = f"{query}, {city}"

    try:
        geocode_result = gmaps.geocode(query)
        # if(geocode_result):
        #     print("Fetched coordinated for: ", query)
        # else:
        #     print("Skipped the query: ", query)
    except Exception as e:
        print("Google Geocoding API error:", e)
        raise

    if not geocode_result:
        print(f"No geocoding result for: {query}")
        return None, None

    location = geocode_result[0]["geometry"]["location"]
    return location["lat"], location["lng"]

def get_distance_matrix(places, city=None):

    # ---------- CLEAN PLACE NAMES ----------
    place_names = []

    for p in places:

        name = p.get("name")

        if name is None:
            continue

        name = name.strip()

        if name == "":
            continue

        # attach city if provided
        if city and city.lower() not in name.lower():
            name = f"{name}, {city}"

        place_names.append(name)

    if len(place_names) == 0:
        raise Exception("No valid places provided")

    # ---------- GOOGLE LIMIT SAFETY ----------
    place_names = place_names[:10]

    try:

        response = gmaps.distance_matrix(
            origins=place_names,
            destinations=place_names,
            mode="driving",
            avoid="ferries"
        )

    except Exception as e:
        print("Google Distance Matrix API error:", e)
        raise

    # ---------- BUILD DISTANCE MATRIX ----------
    matrix = {}

    rows = response["rows"]

    for i, row in enumerate(rows):

        for j, element in enumerate(row["elements"]):

            if i == j:
                continue

            origin = place_names[i]
            destination = place_names[j]

            if element["status"] != "OK":

                print(f"Skipping route {origin} -> {destination}: {element['status']}")

                # fallback large distance so route planner still works
                fallback = {
                    "distance_km": 999,
                    "duration_min": 999,
                    "estimated_cost": 999
                }

                matrix[(origin, destination)] = fallback
                matrix[(destination, origin)] = fallback

                continue

            distance_km = element["distance"]["value"] / 1000
            duration_min = element["duration"]["value"] / 60

            entry = {
                "distance_km": distance_km,
                "duration_min": duration_min,
                "estimated_cost": distance_km * 10
            }

            # store both directions
            matrix[(origin, destination)] = entry
            matrix[(destination, origin)] = entry

    print("Distance matrix size:", len(matrix))

    return matrix