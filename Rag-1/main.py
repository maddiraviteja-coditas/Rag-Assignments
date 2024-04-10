import pandas as pd
from utils import (
    LoadData, 
    SplitData,
    Embedding, 
    PromptTemplate, 
    SimilaritySearch, 
    QA
    )

knolwdge_source = input("""Enter the knolwdge source address : """)

load_knolwdge = LoadData(knolwdge_source)
text = load_knolwdge.load()

print(text)

splitter = SplitData()
chunks = splitter.chunk_data(chunk_size=800, overlap= 200, text= text)

embedding = Embedding()
knolwdge_embedding = embedding.create_embedding(text=chunks)
# knolwdge = pd.read_csv("./embeddings.csv")
# knolwdge_embedding = list(knolwdge["embeddings"])

base_user_prompt = """
Question : {question}
if the question is out of context then say I don't have an idea of the question.
"""

base_system_prompt = """
Based on the context provided answer the question, 
Context: {context}.

"""

prompting = PromptTemplate()
user_prompt = prompting.user_prompt(user_prompt = base_user_prompt)
print(user_prompt)
user_query_embedding = embedding.create_embedding(user_prompt)

similarity_search = SimilaritySearch()
context = similarity_search.similarities(knolwdge_embedding, user_query_embedding, n = 3)
context_list = []
for iter in range(len(context)):
    print(chunks[iter]) 
    context_list.append(chunks[iter])


system_prompt = prompting.system_prompt(system_prompt=base_system_prompt, context= context_list)
print(system_prompt)
qa_model = QA()

response = qa_model.call_ai(system_prompt=system_prompt, user_prompt=user_prompt)

print(response.choices[0].message.content)