import requests
import json
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIRECTORY = "./data"
DATA_SOURCE_URL = "https://raw.githubusercontent.com/mahm/softwaredesign-llm-application/main/02/events_2023.json"
COLLECTION_NAME = "events_2023"
MODEL_NAME = "text-embedding-ada-002"


def fetch_events(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def format_events(events_dict):
    formatted_events = []
    metadatas = []
    for month, dates in events_dict.items():
        for date, events in dates.items():
            for event in events:
                formatted_event = f"### 2023年{date}\n{event}"
                formatted_events.append(formatted_event)
                metadata = {"source": f"2023年{date}"}
                metadatas.append(metadata)
    return formatted_events, metadatas


def main():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    events_dict = fetch_events(DATA_SOURCE_URL)
    formatted_events, metadatas = format_events(events_dict)
    event_ids = [f"id_{i}" for i, _ in enumerate(formatted_events, 1)]

    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
    finally:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name=MODEL_NAME)
        collection = client.create_collection(
            COLLECTION_NAME, embedding_function=openai_ef)
        collection.add(documents=formatted_events,
                       metadatas=metadatas, ids=event_ids)


if __name__ == "__main__":
    main()
