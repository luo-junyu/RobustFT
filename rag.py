import os
import pandas as pd
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from retrying import retry
from func_timeout import func_set_timeout
from common import *

embed = OpenAIEmbeddings(model="text-embedding-3-small")


class NearestReference:
    def __init__(self, k=4) -> None:
        self.vectorstore = None
        self.selector = None
        self.k = k

    def read_data(self, data_path):
        df = pd.read_csv(data_path)
        examples = [row.to_dict() for _, row in df.iterrows()]
        return examples

    def embed_data_path(self, data_path, embed_path=None):
        data = self.read_data(data_path)
        return self.embed_data(data, embed_path)

    def embed_data(self, data, embed_path=None):
        embed_path = embed_path or 'tmp/embed'
        # if os.path.exists(embed_path):
        #     self.vectorstore = FAISS.load_local(embed_path, embed)
            # , allow_dangerous_deserialization=True)
        # else:
        os.makedirs(embed_path, exist_ok=True)
        data_str = [row['question'] for row in data]
        self.vectorstore = FAISS.from_texts(data_str, embed, metadatas=data)
        self.vectorstore.save_local(embed_path)

        self.selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore, k=self.k
        )
        return self.vectorstore

    @retry
    @func_set_timeout(2)
    def retrieve(self, question):
        res = self.selector.select_examples({'question': question})
        return res

    def fewshot(self, question):
        ref = self.retrieve(question['question'])
        ref_str = ''
        for i, r in enumerate(ref):
            ref_str += f"Example {i+1}:\n{format_question_and_answer(r)}\n\n"
        return ref_str
