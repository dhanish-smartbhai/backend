import os
import json
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.tools import tool

load_dotenv()

def get_flights(departure_id, arrival_id, departure_date):
    params = {
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": os.getenv("SEARCH_ENGINE"),
        "hl": os.getenv("LANGUAGE"),                      #language
        "gl": os.getenv("COUNTRY"),                       #country
        "currency": os.getenv("CURRENCY"),
        "no_cache": True,
        "type": os.getenv("FLIGHT_TYPE"),
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": departure_date,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    # Extract both best_flights and other_flights and combine into a single list
    best_flights = results.get("best_flights", [])
    other_flights = results.get("other_flights", [])
    
    # Combine all flights into a single list
    all_flights = best_flights + other_flights

    return all_flights




def fetch_booking_options(booking_token, departure_date, departure_id, arrival_id):
    try:
        params = {
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": os.getenv("SEARCH_ENGINE"),
        "hl": os.getenv("LANGUAGE"),
        "gl": os.getenv("COUNTRY"),   
        "currency": os.getenv("CURRENCY"),
        "type": os.getenv("FLIGHT_TYPE"),
        "no_cache": True,
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": departure_date,
        "booking_token": booking_token,
        "show_hidden": "true",
        "deep_search": "true",
    }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        return results
    except Exception as e:
        print(f"Error fetching booking options for token {booking_token}: {e}")
        return None



@tool
def get_flight_with_aggregator(departure_id: str, arrival_id: str, departure_date: str) -> str:
    """
    Get flight information with aggregated booking options from multiple airlines.
    
    Args:
        departure_id: Departure airport code (e.g., 'DEL' for Delhi)
        arrival_id: Arrival airport code (e.g., 'MAA' for Chennai)  
        departure_date: Departure date in YYYY-MM-DD format
    
    Returns:
        JSON string containing flight data and booking options for each flight
    """
    print("Getting flight with aggregator")
    print(departure_id, arrival_id, departure_date)
    booking_tokens = []
    enhanced_flights = []
    all_flights = get_flights(departure_id, arrival_id, departure_date)

    for flight in all_flights:
        booking_tokens.append(flight.get("booking_token"))

    for booking_token in booking_tokens:
        booking_options = fetch_booking_options(booking_token, departure_date, departure_id, arrival_id)
        enhanced_flights.append({
            "flight_data": booking_options.get("selected_flights")[0].get("flights"),
            "booking_options": booking_options.get("booking_options"),
        })

    return json.dumps(enhanced_flights, indent=2)


# res = get_flight_with_aggregator("DEL", "MAA", "2025-10-05")
# print(res)