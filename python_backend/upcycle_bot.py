from openai import OpenAI
import os
import json
import requests

OPEN_API_KEY = os.environ.get('OPEN_API_KEY')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
client = OpenAI(api_key=OPEN_API_KEY)

def get_upcycling_ideas(item_name, location, money, time):
    prompt= f"""Provide upcycling ideas for a {item_name} considering the following preferences:
    - Location the user lives in: {location} 
    - Whether the user prefers to put in extra money: {money} 
    - The amount of time the user prefers to put in: {time}
    For each idea, include:
    - A unique title
    - A short description
    - Keywords for searching related videos and tutorials
    Output the ideas as a JSON array."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """
                You are an expert upcycling idea generator. Your task is to take the name of an item and generate structured upcycling ideas for it considering the user's preferences i.e 
                Incorporate location-specific suggestions if applicable, suggest ideas that are cost-effective or require money based on the user's preference and suggest ideas that match the user's time availability.
                Each idea should be presented as a JSON object with the following fields:
                - title: A unique and descriptive title for the idea.
                - description: A brief explanation of the idea.
                - keywords: A list of keywords relevant to the idea for searching tutorials or related resources.
                GIVE ONLY THE JSON OBJECT AS THE OUTPUT. YOU DONT HAVE TO OUTPUT A WORD BEFORE OR AFTER. THAT INCLUDES ACCEPTING THE PROMPT AND EVERYTHING.ALSO COMPLETE THE ENTIRE JSON OBJECT INCLUDING THE ] AND } IN THE END.
                Example output:
                [
                    {
                        "title": "DIY Mason Jar Lamp",
                        "description": "Turn an old mason jar into a decorative lamp using LED lights and basic tools.",
                        "keywords": ["mason jar DIY", "lamp tutorial", "upcycling mason jars"]
                    },
                    {
                        "title": "T-Shirt Tote Bag",
                        "description": "Transform an old T-shirt into a reusable tote bag for shopping or storage.",
                        "keywords": ["T-shirt upcycling", "DIY tote bag", "recycling clothes"]
                    }
                ]
                """},
            {"role": "user", "content": prompt}
        ],
       
        max_tokens=300
    )
    ideas = response.choices[0].message.content.strip()

    try: 
        ideas = json.loads(ideas)
    except json.JSONDecodeError:
        return {"error": "Failed to parse GPT-4 response as JSON."}
    
    for idea in ideas:
        idea["videos"] = fetch_youtube_videos(idea["keywords"])
    return ideas

def fetch_youtube_videos(keywords):
    # Use the YouTube API to fetch related videos
    query = " ".join(keywords)
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=3&q={query}&type=video&key={YOUTUBE_API_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        videos = [
            {
                "title": item["snippet"]["title"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in data.get("items", [])
        ]
        return videos
    else:
        return []