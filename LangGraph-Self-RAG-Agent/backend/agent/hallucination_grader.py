from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class HallucinationGrader:
    def __init__(self, hallucination_grader_llm='gpt-4o-mini'):
        # LLM with function call
        self.llm = ChatOpenAI(model=hallucination_grader_llm, temperature=0)
        self.structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        self.system_prompt = """
        You are a grader assessing whether an LLM generation is grounded in \n
        supported by a set of retrieved facts.\n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in \n
        supported by the set of facts."""

    def get_hallucination_grader(self):
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | self.structured_llm_grader
        return hallucination_grader