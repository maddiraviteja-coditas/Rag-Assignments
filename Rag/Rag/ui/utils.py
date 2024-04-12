import PyPDF2
import numpy as np
from dotenv import load_dotenv
import os
import openai
import requests
import re
from ast import literal_eval
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from bs4 import BeautifulSoup

load_dotenv()

class LoadData:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding

    def load_pdf(self):
        pdf = PyPDF2.PdfReader(self.path)
        page = pdf.pages
        text = ""
        for content in page:
            text += (content.extract_text())
        return text
    def load_text(self):
        text = ""
        with open(self.path, "r") as file:
            text += file.read()
        return text
    
    def load_web(self, url: str):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        return text
        
    def get_youtube_text(self, url: str):
        url_attributes_dict = {}
        url_attributes = url.split("?")
        url_query_attributes = url_attributes[-1].split("&")
        for attribute in url_query_attributes:
            dict_attributes = attribute.split("=")
            url_attributes_dict[dict_attributes[0]] = dict_attributes[1] 
        json_text = YouTubeTranscriptApi.get_transcript("tIeHLnjs5U8")
        text = ""
        for dict in json_text:
            text += dict["text"]
        print("sucessfully fetched the text...")
        return text
    
    def load(self):
        print(f"data type : {type(self.path)}")
        check_url = CheckUrl()
        print(self.path)
        if str(self.path).endswith(".txt"):
            return self.load_text()
        elif str(self.path).endswith(".pdf"):
            return self.load_pdf()
        elif check_url.is_url(self.path):
            if check_url.is_youtube_url(self.path):
                print("loading youtube....")
                return self.get_youtube_text(self.path)
            else:
                print("loading website data....")
                return self.load_web(self.path)
        elif type(self.path) == str:
            return self.path
        else:
            print("unable to load data...")


class CheckUrl:
    def __init__(self):
        pass
    def is_url(self, string: str):
        url_pattern = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'  # domain...
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(re.match(url_pattern, string))
    
    def is_youtube_url(self, url: str):
        domain = url.split("?")[0].split(".")
        domain = "youtube" in domain
        return domain


class SplitData:
    def __init__(self):
        pass
    def chunk_data(self,text, chunk_size, overlap):
        chunks = []
        for iteration in range(0, len(text) - chunk_size + 1, chunk_size - overlap):
            chunks.append(text[iteration:iteration+chunk_size])
        print("chunks : " ,chunks)
        return chunks
    
class Embedding:
    def __init__(self, model = "text-embedding-ada-002", api_key = os.getenv("OPENAI_API_KEY")):
        self.model = model
        self.api_key = api_key
        self.data_frame = None
        os.environ["OPENAI_API_KEY"] = str(self.api_key)
        load_dotenv()

    def save_embedding(self, knolwdge_embeddings: list[list[float]], knolwdge_text: list[str]): 
        dictionary = {}
        dictionary["knolwdge_text"] = knolwdge_text
        dictionary["embeddings"] = knolwdge_embeddings
        self.data_frame = pd.DataFrame(dictionary)
        # if not os.path.exists("embeddings\embedding.csv"):
        #     print("tried to create dir")
        #     os.makedirs(directory)
        self.data_frame.to_csv(r"ui\embeddings\embedding_git.csv", index=False)
        return True

    def get_stored_embeddings(self, path: str):
        df = pd.read_csv(path)
        str_embeddings = list(df["embeddings"])
        knolwdge_text = df["knolwdge_text"]
        knolwdge_text = list(knolwdge_text)
        list_embeddings = []
        for embedding in str_embeddings:
            list_embeddings.append(literal_eval(embedding))
        print("Knolwdge_type: ", type(list_embeddings))
        print("text_type: ", type(knolwdge_text))
        return [list_embeddings, knolwdge_text]

    def create_embedding(self, text):
        embedding = openai.OpenAI().embeddings
        embeddings = []
        if type(text) == list:
            for chunk in text:
                chunk_embedding = embedding.create(input=chunk, model = self.model)
                embeddings.append(chunk_embedding.data[0].embedding)
            
            self.save_embedding(embeddings,chunk)
            print("saved embeddings")
            return embeddings
        
        elif type(text) == str:
            chunk_embedding = embedding.create(input = text,  model = self.model)
            input_embedding = chunk_embedding.data[0].embedding
            return input_embedding
        else:
            print("please give the data in list or str format only.")



class PromptTemplate:
    def __init__(self):
        pass

    def user_prompt(self, user_prompt, user_query):
        prompt = user_prompt.format(question = user_query)
        # print(prompt)
        return prompt

    def system_prompt(self, system_prompt, context):
        prompt = system_prompt.format(context = context)
        # print(prompt)
        return prompt
    

class SimilaritySearch:
    def __init__(self):
        pass
    def get_max_index(self,array):
        indexs = []
        for value in array:
            print(np.where(array == value)[0][0])
            indexs.append(np.where(array == value)[0][0])
        return indexs
    
    def similarities(self, knolwdge_vectors, user_query_vecotrs, n = 2):
        dataset = np.array(knolwdge_vectors)
        query_vector = np.array(user_query_vecotrs)
        print("knolwdge Array dim: ",dataset.ndim)
        print("query Array dim: ",query_vector.ndim)
        cos_similarities = cosine_similarity([query_vector], dataset)
        flat_cos_similarities = cos_similarities.flatten()
        print(flat_cos_similarities)
        max_values = flat_cos_similarities.argsort()[-n:]
        index = self.get_max_index(max_values)
        print(index)
        return index

class QA:
    def __init__(self, api_key = os.getenv("OPENAI_API_KEY"), inference_model = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.inference_model = inference_model
        os.environ["OPENAI_API_KEY"] = self.api_key
        load_dotenv()

    def call_ai(self, system_prompt, user_prompt):
        response = openai.chat.completions.create(
            model=self.inference_model,
            messages= [
                    {"role":"system","content" : system_prompt},
                    {"role":"user", "content" : user_prompt}
            ],
            temperature = 1.25
        )
        return response
    



def save_file_to_directory(file, directory):
    try:
        if not os.path.exists(directory):
            print("tried to create dir")
            os.makedirs(directory)
        print("created dir")
        filename = os.path.join(directory, file.name)
        with open(filename, 'wb+') as destination:
            print("file is open...")
            for chunk in file.chunks():
                print("getting_chunks")
                destination.write(chunk)
            print("completed writing file...")

        return True, filename 
    except Exception as e:
        print("got an exception")
        return False, str(e)  


