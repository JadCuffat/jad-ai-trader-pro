import requests

BOT_TOKEN = "7960367964:AAE9_QQx1j44BTnk6v43J2HAvpBGYNzPG-g"
CHAT_ID = "6747508723"  # Update if needed

message = "🚀 Telegram is working correctly!"

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": message}
response = requests.post(url, data=payload)

print(response.text)

