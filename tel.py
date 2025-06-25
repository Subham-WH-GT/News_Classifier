from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest 
from dotenv import load_dotenv
import os

load_dotenv()

# Your API credentials
api_id = os.getenv("api_id") 
api_hash = os.getenv("api_hash")
phone = os.getenv("phone")  # Needed only for first-time login

# Create the client
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start()
    
    # Replace with public channel username (e.g., 'bbcnews')
    target_channel = '@hindustantimes'
    
    entity = await client.get_entity(target_channel)

    history = await client(GetHistoryRequest(
        peer=entity,
        limit=10,         # number of latest messages   
        offset_date=None,
        offset_id=0,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0
    ))

    for message in history.messages:
        if message.message:  # Make sure it's not empty
            print(f" {message.date} - {message.message}\n")

with client:
    client.loop.run_until_complete(main())