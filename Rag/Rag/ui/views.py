from django.shortcuts import render
import openai
from django.http import HttpResponseBadRequest
from ast import literal_eval
from .utils import (LoadData, 
                    Embedding,
                    SplitData,
                    PromptTemplate,
                    SimilaritySearch,
                    QA,
                    save_file_to_directory)
import time
# Create your views here.
EMBEDDINGS = ""
CHUNKS = ""

def home(request):
    return render(request,"home.html")

def load_rag(requests):
    if requests.method == "POST":
        pass
    else:
        return render()
    

def gpt(requests):
    if requests.method == "POST":
        req = requests.POST
        previous_chats = req.get("previous_data")
        query = req.get("query")
        print("query : ", query)
        print("query complete")
        response = openai.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [{"role" : "user", "content" : query}]
        )
        print(response)
        data = {
            "conversation_history" : previous_chats,
            "response" : response.choices[0].message.content
        }
        print(data)
        return render(requests, "chat.html",{"data" : data})
    else:
        return render(requests, "chat.html")


def rag(request):
    if request.method == "POST":
        global EMBEDDINGS, CHUNKS
        req = request.POST
        knolwdge_source = ""
        upload_method = req.get("upload_method")
        print(upload_method)
        if upload_method == "file":
            print("got the file")
            knolwdge_source = request.FILES.get("file")
            save_file_to_directory(knolwdge_source, "ui/uploaded_files/files/")
            time.sleep(10)
            print("file_saved ...")
        elif upload_method == "url":
            print("got the url")
            knolwdge_source = req.get("url")
        elif upload_method == "text":
            print("got the text")
            knolwdge_source = req.get("text")
        elif upload_method == "youtube":
            print("got the video")
            knolwdge_source = req.get("youtube_url")
        elif upload_method == "embeddings":
            print("got the embeddings")
            knolwdge_source = request.FILES.get("embeddings")
            print("step 1")
            save_file_to_directory(knolwdge_source, "ui/uploaded_files/embeddings/")
            time.sleep(10)
            print("saved embeddings...")
        else:
            return HttpResponseBadRequest(content="Bad Response please check...")
        # print(knolwdge_source)
        embedder = Embedding()
        if bool(knolwdge_source):
            print("loading embeddings...")
            if upload_method == "embeddings":
                EMBEDDINGS = None
                path = f"ui/uploaded_files/embeddings/{knolwdge_source}"
                # print(path)
                stored_embeddings = embedder.get_stored_embeddings(path=path)
                # print(type(stored_embeddings[0]))
                EMBEDDINGS = (stored_embeddings[0])
                print("loaded_embeddings")
                CHUNKS = (stored_embeddings[1])
                print(len(CHUNKS))
                print("loaded_chunks")

            elif upload_method == "file":
                path = f"ui/uploaded_files/files/{knolwdge_source}"
                print(path)
                knolwdge_loader = LoadData(path=path)
                knolwdge = knolwdge_loader.load()
            else:
                knolwdge_loader = LoadData(path=knolwdge_source) 
                knolwdge = knolwdge_loader.load()

            
            if upload_method != "embeddings":
                EMBEDDINGS = None
                CHUNKS = None
                splitter = SplitData()
                chunk_size = req.get("chunk_size")
                overlap = req.get("overlap")
                CHUNKS = splitter.chunk_data(knolwdge,chunk_size=chunk_size, overlap=overlap)
                print(len(CHUNKS))
                EMBEDDINGS = embedder.create_embedding(text=CHUNKS)
                embedder.save_embedding(knolwdge_embeddings=EMBEDDINGS, knolwdge_text=CHUNKS)
            # print("knolwdge length: ",len(knolwdge))
            # print("chunks length: ",len(CHUNKS))
            # print(knolwdge)
            return render(request,"chat.html")
            
    else:
        return render(request, "rag.html")
    
def rag_chat(request):

    if request.method == "POST":
        req = request.POST
        query = req.get("query")
        print("in rag_chat")
        template = PromptTemplate()
        embedding = Embedding()
        similiarity_search = SimilaritySearch()
        ai = QA()
        base_user_prompt = """
            Question : {question}
            if the question is out of context then say I don't have an idea of the question.
        """
        base_system_prompt = """
            Based on the context provided answer the question, 
            Context: {context}.
            """
        
        user_prompt = template.user_prompt(base_user_prompt, query)
        user_query_vector = embedding.create_embedding(user_prompt)
        print(type(user_query_vector))
        print(user_query_vector[10])
        print(type(EMBEDDINGS))
        context_index = similiarity_search.similarities(EMBEDDINGS, user_query_vecotrs=user_query_vector, n = 3)
        context_list = []
        print(CHUNKS)
        print(context_index)
        for iter in context_index:
            context_list.append(CHUNKS[iter])
        print(context_list)
        system_prompt = template.system_prompt(base_system_prompt, context=context_list)
        response = ai.call_ai(system_prompt=system_prompt, user_prompt=user_prompt)
        print(response.choices[0].message.content)
        return render(request, "chat.html", {"data":response.choices[0].message.content})
    else:
        return render(request, "chat.html")