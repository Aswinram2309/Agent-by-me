"""
=============================================================================
     AGENTIC AI DEVELOPER - GEMINI LANGCHAIN MULTI-TOOL ReAct AGENT
=============================================================================
Author: Senior Agentic AI Developer
Description: A ReAct (Reasoning and Acting) Multi-Tool Agent using the LangChain
             framework wrapping existing Gemini GenAI function implementations:
             1. Calculator Tool (arithmetic operations: add, subtract, multiply, divide)
             2. Web Search Tool (live web search queries & snippets)
             3. Date & Time Tool (real-time local date and time)
             Integrates LangChain into existing agent logic with minimal changes.
=============================================================================
"""

import datetime
import json
import os
import sys
import urllib.parse
import urllib.request
import warnings
from typing import Any, Dict, List

# Suppress runtime warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Ensure site-packages path is accessible
site_pkg = r"C:\Users\Admin\AppData\Roaming\Python\Python314\site-packages"
if site_pkg not in sys.path:
    sys.path.append(site_pkg)

# LangChain & LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Try importing ddgs or duckduckgo_search safely
try:
    from ddgs import DDGS
    DDG_AVAILABLE = True
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        DDG_AVAILABLE = False


# =====================================================================
# CONFIGURATION
# =====================================================================
API_KEY = "YOUR_GEMINI_API_KEY"
api_key_to_use = os.getenv("GEMINI_API_KEY") if API_KEY == "YOUR_GEMINI_API_KEY" else API_KEY

if not api_key_to_use or api_key_to_use == "YOUR_GEMINI_API_KEY":
    print("\n[!] WARNING: GEMINI_API_KEY is not configured yet in 'agent.py'.")
    print("    Please set API_KEY = 'your_actual_key' inside agent.py or export GEMINI_API_KEY.\n")

MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite"
]


# =====================================================================
# TOOL 1: CALCULATOR TOOL SCHEMA & IMPLEMENTATION
# =====================================================================
CALCULATOR_TOOL_DEFINITION = {
    "name": "calculator",
    "description": "Performs basic arithmetic operations: addition, subtraction, multiplication, and division.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "operation": {
                "type": "STRING",
                "description": "The arithmetic operation to execute.",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "NUMBER",
                "description": "The first number."
            },
            "b": {
                "type": "NUMBER",
                "description": "The second number."
            }
        },
        "required": ["operation", "a", "b"]
    }
}


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Performs basic arithmetic operations: addition, subtraction, multiplication, and division.
    
    Args:
        operation: The arithmetic operation to execute ('add', 'subtract', 'multiply', 'divide').
        a: The first number.
        b: The second number.
        
    Returns:
        JSON string containing operation details, result, and execution status.
    """
    try:
        op = str(operation).lower().strip()
        num_a = float(a)
        num_b = float(b)
        
        if op in ["add", "+"]:
            res = num_a + num_b
        elif op in ["subtract", "-"]:
            res = num_a - num_b
        elif op in ["multiply", "*"]:
            res = num_a * num_b
        elif op in ["divide", "/"]:
            if num_b == 0:
                tool_response = {
                    "status": "error",
                    "operation": op,
                    "a": num_a,
                    "b": num_b,
                    "error": "Division by zero is undefined."
                }
                return json.dumps(tool_response)
            res = num_a / num_b
        else:
            tool_response = {
                "status": "error",
                "operation": op,
                "a": num_a,
                "b": num_b,
                "error": f"Invalid operation '{op}'. Supported operations are add, subtract, multiply, divide."
            }
            return json.dumps(tool_response)

        if res.is_integer():
            res = int(res)

        tool_response = {
            "status": "success",
            "operation": op,
            "a": num_a,
            "b": num_b,
            "result": res
        }
        return json.dumps(tool_response)
    except Exception as e:
        tool_response = {
            "status": "error",
            "operation": operation,
            "a": a,
            "b": b,
            "error": f"Evaluation error: {str(e)}"
        }
        return json.dumps(tool_response)


# =====================================================================
# TOOL 2: WEB SEARCH TOOL SCHEMA & IMPLEMENTATION
# =====================================================================
WEB_SEARCH_TOOL_DEFINITION = {
    "name": "web_search",
    "description": "Searches the internet for live real-time content and retrieves web search result snippets.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search keywords or query string to look up on the web."
            }
        },
        "required": ["query"]
    }
}


@tool
def web_search(query: str) -> str:
    """Searches the internet for live real-time content and retrieves web search result snippets.
    
    Args:
        query: The search keywords or query to look up on the web.
        
    Returns:
        JSON formatted string containing query details, search results list, and execution status.
    """
    clean_query = str(query).strip()
    
    if DDG_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ddgs = DDGS()
                raw_results = list(ddgs.text(clean_query, max_results=5))
            if raw_results:
                results_list = []
                for idx, r in enumerate(raw_results, start=1):
                    results_list.append({
                        "id": idx,
                        "title": r.get("title", "No Title"),
                        "url": r.get("href", "#"),
                        "snippet": r.get("body", "No Description")
                    })
                tool_response = {
                    "status": "success",
                    "query": clean_query,
                    "count": len(results_list),
                    "results": results_list
                }
                return json.dumps(tool_response)
        except Exception:
            pass

    try:
        encoded_query = urllib.parse.quote(clean_query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            heading = data.get("Heading", "")
            
            related_topics = data.get("RelatedTopics", [])
            results_list = []
            if abstract:
                results_list.append({"title": heading, "snippet": abstract})
                
            for topic in related_topics[:4]:
                if isinstance(topic, dict) and "Text" in topic:
                    results_list.append({"snippet": topic["Text"]})

            if results_list:
                tool_response = {
                    "status": "success",
                    "query": clean_query,
                    "count": len(results_list),
                    "results": results_list
                }
                return json.dumps(tool_response)
            else:
                tool_response = {
                    "status": "no_results",
                    "query": clean_query,
                    "message": "No instant answer abstract was returned."
                }
                return json.dumps(tool_response)
    except Exception as e:
        tool_response = {
            "status": "error",
            "query": clean_query,
            "error": str(e)
        }
        return json.dumps(tool_response)


# =====================================================================
# TOOL 3: DATE & TIME TOOL SCHEMA & IMPLEMENTATION
# =====================================================================
DATETIME_TOOL_DEFINITION = {
    "name": "get_current_datetime",
    "description": "Retrieves the current real-time local date and time.",
    "parameters": {
        "type": "OBJECT",
        "properties": {},
        "required": []
    }
}


@tool
def get_current_datetime() -> str:
    """Retrieves the current local real-time date and time formatted in JSON.
    
    Returns:
        JSON formatted string containing current date (D/M/YYYY), current time (HH:MM), and combined string.
    """
    now = datetime.datetime.now()
    formatted_date = f"{now.day}/{now.month}/{now.year}"
    formatted_time = now.strftime("%H:%M")
    full_datetime = f"{formatted_date} {formatted_time}"
    tool_response = {
        "status": "success",
        "current_date": formatted_date,
        "current_time": formatted_time,
        "datetime_formatted": full_datetime,
        "day_of_week": now.strftime("%A")
    }
    return json.dumps(tool_response)


# =====================================================================
# TOOL 4: DOCUMENT SEARCH TOOL SCHEMA & IMPLEMENTATION
# =====================================================================
DOCUMENT_SEARCH_TOOL_DEFINITION = {
    "name": "document_search",
    "description": "Searches local documents and files (txt, md, json, py, csv, log) for matching keywords or content snippets.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search keyword, phrase, or topic to search for inside local documents."
            },
            "folder_path": {
                "type": "STRING",
                "description": "Optional directory path to search within (defaults to current project directory '.')."
            }
        },
        "required": ["query"]
    }
}


@tool
def document_search(query: str, folder_path: str = ".") -> str:
    """Searches local documents and files (txt, md, json, py, csv, log) for matching keywords or content snippets.
    
    Args:
        query: The search keyword, phrase, or topic to search for inside local documents.
        folder_path: Optional directory path to search within (defaults to current project directory '.').
        
    Returns:
        JSON formatted string containing matching file paths, snippets, match count, and execution status.
    """
    clean_query = str(query).strip().lower()
    target_dir = str(folder_path).strip() if folder_path else "."
    
    if not os.path.exists(target_dir):
        return json.dumps({
            "status": "error",
            "query": query,
            "folder": target_dir,
            "error": f"Directory '{target_dir}' does not exist."
        })

    matches = []
    supported_extensions = {".txt", ".md", ".json", ".py", ".csv", ".log", ".html", ".css", ".js"}
    ignore_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode"}

    try:
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            for line_num, line in enumerate(lines, start=1):
                                if clean_query in line.lower():
                                    matches.append({
                                        "file": os.path.relpath(file_path, target_dir),
                                        "line_number": line_num,
                                        "snippet": line.strip()
                                    })
                                    if len(matches) >= 15:
                                        break
                    except Exception:
                        continue
                if len(matches) >= 15:
                    break

        if matches:
            return json.dumps({
                "status": "success",
                "query": query,
                "count": len(matches),
                "matches": matches
            })
        else:
            return json.dumps({
                "status": "no_results",
                "query": query,
                "message": f"No document matches found for '{query}' in '{target_dir}'."
            })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "query": query,
            "error": str(e)
        })


# =====================================================================
# TOOL 5: STUDENT DATABASE RAG SEARCH TOOL SCHEMA & IMPLEMENTATION
# =====================================================================
STUDENT_RAG_TOOL_DEFINITION = {
    "name": "student_rag_search",
    "description": "Retrieves student academic records, marks (Python, DSA, AI, Overall), attendance, placement status, contact details, and department info from the Student Database via RAG retrieval.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search query, student name, student ID, department (CSE, IT, ECE, EEE, AIDS), or criteria (e.g. highest mark, attendance below 75%)."
            }
        },
        "required": ["query"]
    }
}


@tool
def student_rag_search(query: str) -> str:
    """Retrieves student academic records, marks, attendance, placement status, contact details, and department info from the Student Database via RAG retrieval.
    
    Args:
        query: The search query, student name, student ID, department, or criteria.
        
    Returns:
        JSON formatted string containing retrieved student records and match status.
    """
    db_file = os.path.join(os.path.dirname(__file__), "student_database.json")
    if not os.path.exists(db_file):
        return json.dumps({"status": "error", "error": "student_database.json not found."})

    try:
        with open(db_file, "r", encoding="utf-8") as f:
            students = json.load(f)

        clean_q = str(query).strip().lower()

        # RAG Search Strategy: exact or semantic keyword filter
        matched_students = []
        for s in students:
            searchable_text = f"{s['student_id']} {s['name']} {s['dept']} {s['email']} {s['phone']} {s['placement']}".lower()
            if any(word in searchable_text for word in clean_q.split() if len(word) > 2):
                matched_students.append(s)

        # If query is broad (e.g., highest mark, all CSE students, placement eligible, attendance below 75%), return full database context for RAG synthesis
        if not matched_students or any(kw in clean_q for kw in ["all", "highest", "below", "above", "eligible", "attendance", "mark", "department", "who", "which", "how many", "count"]):
            matched_students = students

        return json.dumps({
            "status": "success",
            "query": query,
            "total_records_retrieved": len(matched_students),
            "students": matched_students
        })
    except Exception as e:
        return json.dumps({"status": "error", "query": query, "error": str(e)})


# List of active LangChain tools
LANGCHAIN_TOOLS = [calculator, web_search, get_current_datetime, document_search, student_rag_search]


def extract_text_from_content(content: Any) -> str:
    """Extracts clean text string from LangChain content structures."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    return str(content)


# =====================================================================
# LANGCHAIN ReAct AGENT ENGINE (INTEGRATED INTO EXISTING FLOW)
# =====================================================================
def run_agent_query(user_query: str, active_key: str) -> Dict[str, Any]:
    """Processes any user query using Gemini LLM strictly following the ReAct pattern in LangChain.
    
    Args:
        user_query: Input query (math, web search, date/time, or multi-step combination).
        active_key: Gemini API Key.
        
    Returns:
        Structured dictionary containing user query, ReAct steps (Thought/Action/Observation), and final answer.
    """
    os.environ["GEMINI_API_KEY"] = active_key
    os.environ["GOOGLE_API_KEY"] = active_key

    system_instruction = (
        "You are an intelligent multi-tool AI Agent operating strictly on the ReAct (Reasoning and Acting) pattern.\n"
        "Follow these steps for every user query:\n"
        "1. Thought: Reason about the user's intent and decide which tool (calculator, web_search, get_current_datetime, document_search, or student_rag_search) is needed.\n"
        "2. Action: Call the appropriate tool:\n"
        "   - 'student_rag_search' for student database queries (marks, attendance, placement eligibility, student details, departments).\n"
        "   - 'get_current_datetime' for date and time requests.\n"
        "   - 'calculator' for arithmetic operations (with operation, a, and b).\n"
        "   - 'web_search' for live web information and real-time research.\n"
        "   - 'document_search' for searching keywords inside local project files and documents.\n"
        "   If the query asks for multiple tasks, execute the required tools in logical sequential order.\n"
        "3. Observation: Evaluate the JSON result returned by the tool.\n"
        "4. Final Answer: Provide the final accurate answer to the user based on your reasoning and observation.\n"
        "When answering date and time queries, state the date and time clearly using the format D/M/YYYY HH:MM (e.g., 8/10/2026 11:07)."
    )

    last_exception = None
    successful_model = None
    agent_app = None
    messages = []

    for model_name in MODEL_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=active_key,
                temperature=0.1,
                max_retries=2
            )
            agent_app = create_react_agent(llm, tools=LANGCHAIN_TOOLS, prompt=system_instruction)
            
            inputs = {"messages": [HumanMessage(content=user_query)]}
            response_state = agent_app.invoke(inputs)
            messages = response_state.get("messages", [])
            successful_model = model_name
            break
        except Exception as err:
            last_exception = err

    if agent_app is None or successful_model is None:
        raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_exception}")

    # Build ReAct Trajectory (Thought -> Action -> Observation)
    react_steps = []
    tool_used = False
    final_answer = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_used = True
                for tc in msg.tool_calls:
                    thought_text = extract_text_from_content(msg.content)
                    if not thought_text:
                        thought_text = f"To answer '{user_query}', I should invoke the '{tc.get('name')}' tool."
                    react_steps.append({
                        "step": "Thought",
                        "content": thought_text
                    })
                    react_steps.append({
                        "step": "Action",
                        "tool": tc.get("name"),
                        "arguments": tc.get("args")
                    })
            elif msg.content:
                final_answer = extract_text_from_content(msg.content)
        elif isinstance(msg, ToolMessage):
            react_steps.append({
                "step": "Observation",
                "content": extract_text_from_content(msg.content)
            })

    if not final_answer and messages:
        final_answer = extract_text_from_content(messages[-1].content)

    return {
        "user_query": user_query,
        "model_used": successful_model,
        "tool_used": tool_used,
        "react_steps": react_steps,
        "agent_response": final_answer.strip()
    }


# =====================================================================
# MAIN INTERACTIVE EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("     AGENTIC AI DEVELOPER - GEMINI ReAct MULTI-TOOL AGENT")
    print("     (POWERED BY LANGCHAIN FRAMEWORK)")
    print("=" * 70)
    print("Framework: LangChain Framework (ChatGoogleGenerativeAI + LangGraph)")
    print("LLM Model: Google AI Studio Gemini LLM")
    print("Tools:     1. Calculator Tool (@tool)")
    print("           2. Web Search Tool (@tool)")
    print("           3. Date & Time Tool (@tool)")
    print("           4. Document Search Tool (@tool)")
    print("           5. Student Database RAG Search Tool (@tool)")
    print("=" * 70)

    # Validate API key
    if not api_key_to_use or api_key_to_use == "YOUR_GEMINI_API_KEY":
        user_key = input("\nEnter your Google AI Studio Gemini API Key: ").strip()
        if not user_key:
            print("\n[X] Error: API key is required to run the agent. Exiting.")
            return
        active_api_key = user_key
    else:
        active_api_key = api_key_to_use

    first_run = True

    try:
        while True:
            print("\nProvide any question for the Agent (Math, Web Search, Date & Time, or Multi-Task).")
            if first_run:
                print("(Press Enter to use default sample query, or type 'exit' to quit)\n")
            else:
                print("(Type 'exit' or 'quit' to stop)\n")

            user_input = input("Enter Query: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting Gemini ReAct Agent. Goodbye!")
                break

            if not user_input:
                if first_run:
                    user_input = "What is today's date and time?"
                    print(f"  (Using default sample query: '{user_input}')")
                else:
                    print("No query provided. Type 'exit' to quit.")
                    continue

            first_run = False

            print("\nProcessing with LangChain Agent...\n")

            result = run_agent_query(user_input, active_api_key)

            # Display ReAct Workflow Trajectory
            if result.get("react_steps"):
                print("=" * 70)
                print("                       AGENT WORKFLOW (ReAct)")
                print("=" * 70)
                for step in result["react_steps"]:
                    if step["step"] == "Thought":
                        print(f"[Thought] {step['content']}")
                    elif step["step"] == "Action":
                        print(f"[Action] Calling tool '{step['tool']}' with arguments: {step['arguments']}")
                    elif step["step"] == "Observation":
                        print(f"[Observation] Returned: {step['content']}")
                print("=" * 70 + "\n")

            # Display final answer
            print("=" * 70)
            print("                       FINAL AGENT ANSWER")
            print("=" * 70)
            print(result["agent_response"])
            print("=" * 70)

            # Prepare structured JSON summary payload
            output_data = {
                "status": "success",
                "framework": "LangChain",
                "agent_name": "Gemini ReAct Multi-Tool Agent (LangChain)",
                "tools_available": ["calculator", "web_search", "get_current_datetime", "document_search", "student_rag_search"],
                "results": result
            }

            # Save output to JSON file
            output_filename = "output.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            print(f"\n[+] Results successfully saved to '{output_filename}'")
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting Gemini ReAct Agent. Goodbye!")


if __name__ == "__main__":
    main()
