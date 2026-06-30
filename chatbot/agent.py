import os
from typing import TypedDict, List, Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

class ExtractedIssue(BaseModel):
    fault_code: Optional[str] = Field(description="The fault code if mentioned (e.g. F5001)")
    symptoms: List[str] = Field(description="List of symptoms described by the user")
    components: List[str] = Field(description="List of specific components mentioned (e.g. cooling fan, DC bus)")

class DiagnosisEvaluation(BaseModel):
    is_fault_code_complete: bool = Field(description="Is the fault code complete? (e.g. F5001)")
    has_enough_symptoms: bool = Field(description="Are there enough specific symptoms described?")
    has_conflicting_symptoms: bool = Field(description="Are the symptoms conflicting with each other or the fault code?")
    ready_for_diagnosis: bool = Field(description="Can a preliminary diagnosis be made based on the available information?")
    follow_up_instructions: str = Field(description="If additional information would improve confidence, what specific info should the follow-up ask for? (Be empty if not needed).")

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    extracted_issue: ExtractedIssue
    documents: list
    confidence: str
    ready_for_diagnosis: bool
    follow_up_instructions: str
    retrieval_score: float

def get_agent():
    persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    evaluator_llm = llm.with_structured_output(DiagnosisEvaluation)
    extractor_llm = llm.with_structured_output(ExtractedIssue)

    def analyze_and_extract(state: GraphState):
        system_prompt = """You are an expert diagnostic assistant for ABB ACS880 drives. 
Your task is to analyze the conversation history and extract key information.

Extract:
- Fault Code
- Symptoms
- Component names

Normalize technical terminology (e.g., "fan jammed" -> "Cooling Fan Failure").
Do not invent information."""
        
        latest_user = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_user),
        ]
        
        result = extractor_llm.invoke(messages)
        return {"extracted_issue": result}

    def retrieve_cases(state: GraphState):
        issue = state.get("extracted_issue")
        
        fault_code = issue.fault_code if issue and issue.fault_code else "None"
        symptoms = ",".join(issue.symptoms) if issue and issue.symptoms else "None"
        components = ",".join(issue.components) if issue and issue.components else "None"

        latest_user = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        query = f"""Fault Code:
{fault_code}

Symptoms:
{symptoms}

Components:
{components}

Original User Problem:
{latest_user}

Industrial Equipment:
ABB ACS880 Variable Frequency Drive"""

        results = vector_store.similarity_search_with_relevance_scores(query, k=8)
        
        if not results:
            return {"documents": [], "retrieval_score": 0.0}

        docs = [doc for doc, score in results]
        best_score = max([score for _, score in results], default=0.0)
        
        return {
            "documents": docs,
            "retrieval_score": float(best_score)
        }

    def evaluate_diagnosis(state: GraphState):
        docs = state.get("documents", [])
        doc_texts = "\n\n".join([d.page_content for d in docs])
        issue = state.get('extracted_issue')
        
        issue_str = f"Fault Code: {issue.fault_code}\nSymptoms: {','.join(issue.symptoms)}\nComponents: {','.join(issue.components)}" if issue else "None"

        system_prompt = f"""You are a master ABB ACS880 Field Service Engineer.

Conversation Summary:
{issue_str}

Retrieved Historical Cases:
{doc_texts}

Retrieval Similarity Score:
{state.get('retrieval_score', 0)}

You have already retrieved the most relevant historical failures.
Your job is NOT to decide whether more cases are needed.
Only decide whether a preliminary diagnosis can be made.

Evaluate ONLY:
1. Can you provide a preliminary diagnosis based on the available information?
2. Is the fault code complete?
3. Are the symptoms conflicting?

Do not evaluate retrieval quality.

If yes, set ready_for_diagnosis=True.
If additional information would improve confidence, include it in follow_up_instructions, but do not reject diagnosis unless absolutely impossible."""
        
        latest_user = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_user),
        ]
        
        evaluation = evaluator_llm.invoke(messages)
        
        return {
            "confidence": "high" if evaluation.ready_for_diagnosis else "low",
            "ready_for_diagnosis": evaluation.ready_for_diagnosis,
            "follow_up_instructions": evaluation.follow_up_instructions
        }

    def generate_answer(state: GraphState):
        docs = state.get("documents", [])
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Case {i}\n\n{doc.page_content}\n\nSource:\n{doc.metadata}")
        
        doc_texts = "\n\n---\n\n".join(context_parts)

        conversation = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in state["messages"]
        )

        system_prompt = f"""You are an ABB ACS880 maintenance assistant.

Conversation History
{conversation}

--------------------------------

The following are historical maintenance records retrieved from the maintenance database.

==============================
HISTORICAL MAINTENANCE RECORDS
==============================

{doc_texts}

==============================

These records are your ONLY source of truth.

Rules:

1. NEVER answer using your own knowledge.
2. NEVER explain ABB fault codes unless the retrieved records mention them.
3. NEVER infer causes that are not supported by retrieved records.
4. Every diagnosis must be supported by retrieved evidence.
5. If there is insufficient evidence, explicitly state:
   "The historical maintenance records do not provide enough evidence to determine the root cause."

Answer using this structure:

## Retrieved Evidence

## Analysis

## Possible Causes

## Recommended Checks

## Recommended Repair
"""
        
        latest_user = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_user),
        ]
        
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}

    def ask_follow_up(state: GraphState):
        system_prompt = f"""You are a professional Technical Support Engineer for ABB ACS880 drives.
The user is experiencing an issue, but we cannot diagnose it yet.

Missing/Conflicting Info: {state.get('follow_up_instructions', 'We need more details about the drive state or fault codes.')}

Write a polite, professional follow-up response.
Ask at most 3 specific technical questions.
Prioritize:
1. Exact fault code
2. Operating condition
3. Measurements

Keep it concise and supportive based on the user's issue."""
        
        latest_user = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_user),
        ]
        
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}

    def route_grading(state: GraphState):
        retrieval_score = state.get("retrieval_score", 0.0)
        ready_for_diagnosis = state.get("ready_for_diagnosis", False)
        
        if ready_for_diagnosis and retrieval_score >= 0.45:
            return "generate_answer"
        else:
            return "ask_follow_up"

    workflow = StateGraph(GraphState)
    
    workflow.add_node("analyze_and_extract", analyze_and_extract)
    workflow.add_node("retrieve_cases", retrieve_cases)
    workflow.add_node("evaluate_diagnosis", evaluate_diagnosis)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("ask_follow_up", ask_follow_up)

    workflow.add_edge(START, "analyze_and_extract")
    workflow.add_edge("analyze_and_extract", "retrieve_cases")
    workflow.add_edge("retrieve_cases", "evaluate_diagnosis")
    
    workflow.add_conditional_edges(
        "evaluate_diagnosis",
        route_grading,
        {
            "generate_answer": "generate_answer",
            "ask_follow_up": "ask_follow_up"
        }
    )
    
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("ask_follow_up", END)

    app = workflow.compile()
    return app
