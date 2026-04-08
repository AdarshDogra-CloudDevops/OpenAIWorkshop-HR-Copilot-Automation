# Agent class
import openai
import os
from pathlib import Path
import json
import time
import uuid
import inspect
import numpy as np

from azure.search.documents.models import Vector
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from dotenv import load_dotenv

# ================== ENV SETUP ==================
env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
openai.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"

emb_engine = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")
emb_engine = emb_engine.strip('"')


# ================== HELPER FUNCTIONS ==================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text):
    openai.api_version = "2023-05-15"
    response = openai.Embedding.create(
        input=text,
        engine=emb_engine
    )
    return response["data"][0]["embedding"]


# ================== FAISS SEARCH ==================
class Search_Client():
    def __init__(self, emb_map_file_path):
        with open(emb_map_file_path) as file:
            self.chunks_emb = json.load(file)

    def find_article(self, question, topk=3):
        input_vector = get_embedding(question)

        cosine_list = []
        for chunk_id, chunk_content, vector in self.chunks_emb:
            cosine_sim = cosine_similarity(input_vector, vector)
            cosine_list.append((chunk_id, chunk_content, cosine_sim))

        cosine_list.sort(key=lambda x: x[2], reverse=True)
        cosine_list = cosine_list[:topk]

        text_content = ""
        for chunk_id, content, _ in cosine_list:
            text_content += f"{chunk_id}\n{content}\n"

        return text_content


# ================== AZURE SEARCH ==================
if os.getenv("USE_AZCS") == "True":
    service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME").strip('"')
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY").strip('"')

    def generate_embeddings(text):
        return get_embedding(text)

    credential = AzureKeyCredential(key)
    azcs_search_client = SearchClient(service_endpoint, index_name=index_name, credential=credential)

else:
    faiss_search_client = Search_Client("./data/chunk_emb_map.json")


def search_knowledgebase_acs(search_query):
    vector = Vector(value=generate_embeddings(search_query), k=3, fields="contentVector")

    results = azcs_search_client.search(
        search_text=search_query,
        vectors=[vector],
        select=["id", "content"],
        top=5
    )

    text_content = ""
    for result in results:
        text_content += f"{result['id']}\n{result['content']}\n"

    return text_content


def search_knowledgebase_faiss(search_query):
    return faiss_search_client.find_article(search_query)


def search_knowledgebase(search_query):
    if os.getenv("USE_AZCS") == "True":
        return search_knowledgebase_acs(search_query)
    else:
        return search_knowledgebase_faiss(search_query)


# ================== SEMANTIC CACHE ==================
if os.getenv("USE_SEMANTIC_CACHE") == "True":
    cache_index_name = os.getenv("CACHE_INDEX_NAME").strip('"')
    azcs_semantic_cache_search_client = SearchClient(service_endpoint, cache_index_name, credential=credential)


def add_to_cache(search_query, gpt_response):
    search_doc = {
        "id": str(uuid.uuid4()),
        "search_query": search_query,
        "search_query_vector": get_embedding(search_query),
        "gpt_response": gpt_response
    }
    azcs_semantic_cache_search_client.upload_documents(documents=[search_doc])


def get_cache(search_query):
    vector = Vector(value=get_embedding(search_query), k=3, fields="search_query_vector")

    results = azcs_semantic_cache_search_client.search(
        search_text=None,
        vectors=[vector],
        select=["gpt_response"],
    )

    try:
        result = next(results)
        if result['@search.score'] >= float(os.getenv("SEMANTIC_HIT_THRESHOLD")):
            return result['gpt_response']
    except StopIteration:
        pass

    return None


# ================== GPT STREAM ==================
def gpt_stream_wrapper(response):
    for chunk in response:
        chunk_msg = chunk['choices'][0]['delta']
        yield chunk_msg.get('content', "")


# ================== AGENT ==================
class Agent():
    def __init__(self, engine, persona, name=None, init_message=None):
        if init_message:
            self.init_history = [
                {"role": "system", "content": persona},
                {"role": "assistant", "content": init_message}
            ]
        else:
            self.init_history = [{"role": "system", "content": persona}]

        self.persona = persona
        self.engine = engine
        self.name = name

    def generate_response(self, new_input, history=None, stream=False, request_timeout=20, api_version="2023-05-15"):
        openai.api_version = api_version

        if new_input is None:
            return self.init_history[1]["content"]

        messages = self.init_history.copy()

        if history:
            for user_q, bot_r in history:
                messages.append({"role": "user", "content": user_q})
                messages.append({"role": "assistant", "content": bot_r})

        messages.append({"role": "user", "content": new_input})

        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=messages,
            stream=stream,
            request_timeout=request_timeout
        )

        if not stream:
            return response['choices'][0]['message']['content']
        else:
            return gpt_stream_wrapper(response)

    def run(self, **kwargs):
        return self.generate_response(**kwargs)


# ================== ARG CHECK ==================
def check_args(function, args):
    sig = inspect.signature(function)

    for name in args:
        if name not in sig.parameters:
            return False

    for name, param in sig.parameters.items():
        if param.default is param.empty and name not in args:
            return False

    return True


# ================== SMART AGENT ==================
class Smart_Agent(Agent):
    def __init__(self, persona, functions_spec, functions_list, name=None, init_message=None,
                 engine=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")):
        super().__init__(engine=engine, persona=persona, init_message=init_message, name=name)
        self.functions_spec = functions_spec
        self.functions_list = functions_list

    def run(self, user_input, conversation=None, stream=False, api_version="2023-07-01-preview"):
        openai.api_version = api_version

        if user_input is None:
            return self.init_history, self.init_history[1]["content"]

        if conversation is None:
            conversation = self.init_history.copy()

        conversation.append({"role": "user", "content": user_input})
        query_used = None

        response = openai.ChatCompletion.create(
            deployment_id=self.engine,
            messages=conversation,
            functions=self.functions_spec,
            function_call="auto",
        )

        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]

            if function_name not in self.functions_list:
                raise Exception(f"Function {function_name} does not exist")

            function_to_call = self.functions_list[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])

            if not check_args(function_to_call, function_args):
                raise Exception(f"Invalid arguments for {function_name}")

            if function_name == "search_knowledgebase" and os.getenv("USE_SEMANTIC_CACHE") == "True":
                cache_output = get_cache(function_args["search_query"])
                if cache_output:
                    conversation.append({"role": "assistant", "content": cache_output})
                    return False, query_used, conversation, cache_output
                else:
                    query_used = function_args["search_query"]

            function_response = function_to_call(**function_args)

            conversation.append({
                "role": response_message["role"],
                "name": function_name,
                "content": response_message["function_call"]["arguments"],
            })

            conversation.append({
                "role": "function",
                "name": function_name,
                "content": function_response,
            })

            second_response = openai.ChatCompletion.create(
                messages=conversation,
                deployment_id=self.engine,
                stream=stream,
            )

            if not stream:
                assistant_response = second_response["choices"][0]["message"]["content"]
                conversation.append({"role": "assistant", "content": assistant_response})
            else:
                assistant_response = second_response

            return stream, query_used, conversation, assistant_response

        else:
            assistant_response = response_message["content"]
            conversation.append({"role": "assistant", "content": assistant_response})

        return False, query_used, conversation, assistant_response
