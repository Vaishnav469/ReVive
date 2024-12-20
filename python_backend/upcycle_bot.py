from openai import OpenAI
import os

OPEN_API_KEY = os.environ.get('OPEN_API_KEY')
client = OpenAI(api_key=OPEN_API_KEY)

def get_upcycling_ideas(item_name):
    prompt=f"Provide different upcycling ideas for a {item_name}."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a expert Upcycling idea Generator. Take the name of the the item, and generate upcycle ideas for it."},
            {"role": "user", "content": prompt}
        ],
       
        max_tokens=150
    )
    ideas = response.choices[0].message.content.strip()
    return ideas