import PyPDF2
import numpy as np
from dotenv import load_dotenv
import os
import openai
import requests
import re
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
        check_url = CheckUrl()
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
        return chunks
    
class Embedding:
    def __init__(self, model = "text-embedding-ada-002", api_key = os.getenv("OPENAI_API_KEY")):
        self.model = model
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = str(self.api_key)
        load_dotenv()

    def create_embedding(self, text):
        embedding = openai.OpenAI().embeddings
        embeddings = []
        if type(text) == list:
            for chunk in text:
                chunk_embedding = embedding.create(input=chunk, model = self.model)
                embeddings.append(chunk_embedding.data[0].embedding)
            # df = pd.DataFrame(embeddings)
            # df.to_csv("embeddings.csv")
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

    def user_prompt(self, user_prompt):
        user_query = input("What's your query? ")
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
        array = list(array)
        array2 = list(np.copy(array))
        indexs = 0
        max_element = np.max(array2)
        indexs = (array.index(max_element))
        array2.remove(max_element)
        return indexs
    
    def similarities(self, knolwdge_vectors, user_query_vecotrs, n = 2):
        dataset = np.array(knolwdge_vectors)
        query_vector = np.array(user_query_vecotrs)
        cos_similarities = cosine_similarity([query_vector], dataset)
        flat_cos_similarities = cos_similarities.flatten()
        index = []
        for _ in range(n):
            index.append(self.get_max_index(flat_cos_similarities))

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
            ]
        )
        return response