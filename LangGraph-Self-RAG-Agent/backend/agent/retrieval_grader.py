### Retrieval Grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class RetrievalGrader:
    def __init__(self, retrieval_grader_llm='gpt-4o-mini'):
        self.llm = ChatOpenAI(model=retrieval_grader_llm, temperature=0)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.system_prompt ="""
            You are a grader assessing relevance of a retrieved document to a user question. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            """


    def get_retrieval_grader(self):

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | self.structured_llm_grader
        return retrieval_grader
