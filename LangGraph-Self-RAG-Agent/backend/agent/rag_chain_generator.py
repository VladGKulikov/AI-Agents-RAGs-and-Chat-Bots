from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class RagChainGenerator:
    def __init__(self, rag_chain_generator_llm='gpt-4o-mini'):
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.llm = ChatOpenAI(model_name=rag_chain_generator_llm, temperature=0)

    def get_rag_chain(self):
        # Post-processing
        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()

        return rag_chain
