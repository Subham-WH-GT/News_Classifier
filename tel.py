# from telethon.sync import TelegramClient
# from telethon.tl.functions.messages import GetHistoryRequest 
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Your API credentials
# api_id = os.getenv("api_id") 
# api_hash = os.getenv("api_hash")
# phone = os.getenv("phone")  # Needed only for first-time login

# # Create the client
# client = TelegramClient('session_name', api_id, api_hash)

# async def main():
#     await client.start()
    
#     # Replace with public channel username (e.g., 'bbcnews')
#     target_channel = '@hindustantimes'
    
#     entity = await client.get_entity(target_channel)

#     history = await client(GetHistoryRequest(
#         peer=entity,
#         limit=10,         # number of latest messages   
#         offset_date=None,
#         offset_id=0,
#         max_id=0,
#         min_id=0,
#         add_offset=0,
#         hash=0
#     ))

#     for message in history.messages:
#         if message.message:  # Make sure it's not empty
#             print(f" {message.date} - {message.message}\n")

# with client:
#     client.loop.run_until_complete(main()) 




from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest 
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime

load_dotenv()

# Your API credentials
api_id = os.getenv("api_id")
api_hash = os.getenv("api_hash")
phone = os.getenv("phone")  # Needed only for first-time login

# Create the client
client = TelegramClient('session_name', api_id, api_hash)

# Regex to extract links
url_pattern = re.compile(r'https?://\S+')

async def main():
    await client.start()
    
    target_channel = '@hindustantimes'  # Replace with any public channel
    entity = await client.get_entity(target_channel)

    history = await client(GetHistoryRequest(
        peer=entity,
        limit=10,
        offset_date=None,
        offset_id=0,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0
    ))

    news_list = []

    for message in history.messages:
        if message.message:  # Not empty
            date_str = message.date.strftime('%Y-%m-%d %H:%M:%S')
            text = message.message.strip()
            urls = url_pattern.findall(text)
            clean_text = url_pattern.sub('', text).strip()

            news_item = {
                "date": date_str,
                "text": clean_text,
                "urls": urls
            }

            news_list.append(news_item)

    # Save to JSON
    with open("telegram_news.json", "w", encoding="utf-8") as f:
        json.dump(news_list, f, indent=4, ensure_ascii=False)

    print(f" Saved {len(news_list)} messages to telegram_news.json")

with client:
    client.loop.run_until_complete(main())
