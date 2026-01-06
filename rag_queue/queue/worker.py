from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

openai_client = OpenAI()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vector_db = QdrantVectorStore.from_existing_collection(
     url="http://localhost:6333",
     collection_name="learning_rag",
     embedding=embeddings,
)

def process_query_worker(query):
    search_result = vector_db.similarity_search(query=query);

    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile location: {result.metadata['source']}" for result in search_result])

    SYSTEM_PROMPT = f"""
    You are a helpfull AI assistant who answers user query based on the available context retrieved from a pdf file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user to open the right page number to know more

    Context: {context}
    """
    response = openai_client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "system", "content": SYSTEM_PROMPT
        },
        {
            "role": "user", "content": query
        },

    ]
    )
    print(f"🤖: {response.choices[0].message.content}" )
    return response.choices[0].message.content

# use before running rq worker
#export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
