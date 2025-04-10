import requests

api_key = "AIzaSyBOeQxUot1f9QGqfcHXfSaQ0V-n2vC1D1M"
origin = "New York,USA"
destination = "Los Angeles,USA"
url = f"https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix?key={api_key}"

response = requests.get(url)
print(response.json())
