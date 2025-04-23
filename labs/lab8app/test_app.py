import requests

phone = {"ram": 2000, "battery_power": 1200, "px_height": 700, "px_width": 1200}

url = "http://127.0.0.1:8000/predict"
response = requests.post(url, json=phone)
print(response.json())
