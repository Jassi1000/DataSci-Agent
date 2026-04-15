import os
import json
import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PlanStep(BaseModel):
    step_id: int = Field(description="Sequential ID of the step")
    agent: str = Field(description="Name of the agent handling the step (e.g., data_cleaning, eda, ml, evaluation, report)")
    action: str = Field(description="Description of the action to be performed")

class ExecutionPlan(BaseModel):
    problem_type: str = Field(description="Task inference: classification | regression | clustering | unknown")
    steps: List[PlanStep] = Field(description="List of steps in the execution plan")

class PlannerAgent:
    """
    Planner Agent for generating structured execution plans for dataset analysis.
    """
    
    def __init__(self, model_name: str = "llama3-70b-8192", temperature: float = 0.0):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable is not set. API calls may fail.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        self.prompt = self.build_prompt()
        self.chain = self.prompt | self.llm | self.parser

    def build_prompt(self) -> PromptTemplate:
        template = """You are an expert Data Scientist acting as a Planner Agent.
Your task is to generate a structured execution plan for a dataset analysis pipeline based on the provided dataset summary and an optional user goal.

Dataset Summary:
{dataset_summary}

User Goal:
{user_goal}

Instructions:
1. Infer the logical problem type (classification, regression, clustering, or unknown) from the dataset summary and user goal.
2. Formulate logical execution steps involving specific agents (must use any of: data_cleaning, eda, ml, evaluation, report).
3. Adapt the steps depending on the dataset characteristics mentioned in the summary.
4. Output EXACTLY adhering to the JSON format instructions. Do NOT include markdown code blocks, just raw valid JSON.

Format Instructions:
{format_instructions}
"""
        return PromptTemplate(
            template=template,
            input_variables=["dataset_summary", "user_goal"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def generate_plan(self, dataset_summary: Dict[str, Any], user_goal: Optional[str] = None, retries: int = 3) -> dict:
        goal = user_goal if user_goal and user_goal.strip() else "Explore the dataset and build an appropriate baseline ML model."
        
        for attempt in range(retries):
            try:
                logger.info(f"Generating plan. Attempt {attempt + 1}/{retries}")
                
                plan = self.chain.invoke({
                    "dataset_summary": json.dumps(dataset_summary, indent=2),
                    "user_goal": goal
                })
                
                # Enforce structure validation via Pydantic
                valid_plan = ExecutionPlan(**plan)
                logger.info("Plan successfully generated and validated.")
                return valid_plan.model_dump()
                
            except (ValidationError, Exception) as e:
                logger.error(f"Error generating plan on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    logger.error("Max retries reached. Returning default safe plan.")
                    return self._fallback_plan()
                    
        return self._fallback_plan()

    def _fallback_plan(self) -> dict:
        fallback = ExecutionPlan(
            problem_type="unknown",
            steps=[
                PlanStep(step_id=1, agent="data_cleaning", action="Handle missing values and duplicates"),
                PlanStep(step_id=2, agent="eda", action="Perform exploratory data analysis"),
                PlanStep(step_id=3, agent="ml", action="Train initial baseline models"),
                PlanStep(step_id=4, agent="evaluation", action="Evaluate model performance"),
                PlanStep(step_id=5, agent="report", action="Generate findings report")
            ]
        )
        return fallback.model_dump()
