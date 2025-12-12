import os
import uuid
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Dict, Optional, Literal, List, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from  langgraph.prebuilt import ToolNode
import pandas as pd

SYSTEM_PROMPT = """
You are an LLM agent operating on a local typed graph stored in memory.

The graph consists of:
- Node tables (one table per node type): Gene, Anatomy, CellType.
- Edge tables (one table per relation type) with columns:
  src_id, src_type, dst_id, dst_type, rel_type.

You do NOT have direct access to the graph data.
All graph operations MUST be performed ONLY via the provided tools.

Your task is to transform a free-form English user request into a sequence of tool calls that:
1) Determine the ordered node-type path (Type0 -> Type1 -> ... -> TypeN).
2) Select nodes for each node type (optionally filtered by exact feature values).
3) Select relation types per step (or use all if unspecified).
4) Find all paths that match the requested structure.
5) Save the final tabular result to a CSV file.

Available tools:
- resolve_nodes
- find_paths
- save_csv

Rules:
- Always use tools for graph operations.
- Never invent nodes, relations, features, or IDs.
- If node filters are not specified, use ALL nodes of that type.
- If relation types are not specified for a step, use ALL relations for that step.
- Always call save_csv before responding to the user.
- Do not explain internal reasoning.
- Do not output raw data unless explicitly requested.
"""

load_dotenv()

class Graph(TypedDict):
    nodes: Dict[str, pd.DataFrame]
    edges: Dict[str, pd.DataFrame]

# Global in-memory graph
GRAPH: Optional[Graph] = None

def init_global_graph(graph_path: str):
    """
    Initialize the global GRAPH variable from CSV files in `graph_path`.
    This function should be called once at startup or from the first LangGraph node.
    """
    global GRAPH

    base = Path(graph_path)
    if not base.exists():
        raise FileNotFoundError(f"graph_path '{base}' does not exist")

    nodes: Dict[str, pd.DataFrame] = {}
    edges: Dict[str, pd.DataFrame] = {}

    # Load nodes
    for path in base.glob("type=*.csv"):
        # 'type=Gene.csv' -> 'Gene'
        node_type = path.stem.split("=", 1)[1]

        df = pd.read_csv(path)
        nodes[node_type] = df

    # Load edges
    for path in base.glob("rel=*.csv"):
        # 'rel=EXPRESSES_AeG.csv' -> 'EXPRESSES_AeG'
        rel_type = path.stem.split("=", 1)[1]

        df = pd.read_csv(path)
        edges[rel_type] = df

    GRAPH = Graph(nodes=nodes, edges=edges)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def init_graph_node(state: AgentState) -> dict:
    """First node in the StateGraph. Initialize the global GRAPH."""
    init_global_graph("NodesEdges")
    return {}

@tool
def resolve_nodes(
    node_type: Literal["Gene", "Anatomy", "CellType"],
    selection_type: Literal["all", "feature_list"] = "all",
    feature_names: Optional[List[str]] = None,
    values: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Selects a subset of nodes from the global in-memory graph (GRAPH["nodes"]) based on the
    node type and optional feature-based filters.

    This tool is used by the LLM to determine *which specific nodes* should be included
    in the path-finding query. The LLM decides how to filter the nodes, and this tool
    performs the actual filtering.

    PARAMETERS
    ----------
    node_type : {"Gene", "Anatomy", "CellType"}
        The node type to select from. Must match one of the node tables that exist
        in the loaded GRAPH.

    selection_type : {"all", "feature_list"}
        - "all" :
            Return ALL nodes of the specified node_type without any filtering.
        - "feature_list":
            Apply exact-match filtering for one or more node features (columns).

    feature_names : list[str], optional
        A list of column names to filter by. These must be columns present
        in the node dataframe of the specified node_type.
        Example: ["name"], ["name", "chromosome"], etc.

    values : dict[str, list[str]], optional
        For each feature in feature_names, this dict provides the allowed values.
        Only rows where df[feature] is exactly in values[feature] will be kept.

        Example:
        {
            "name": ["DPP4", "GNAS"],
            "chromosome": ["17"]
        }

        Filtering is applied as:
            df = df[df[feature1].isin(values[feature1])]
            df = df[df[feature2].isin(values[feature2])]
        AND logic across all features.

    RETURN
    ------
    list[dict]
        A JSON-serializable list of rows representing the selected nodes.
        Each row contains ALL columns from the node dataframe (no columns are removed).

    IMPORTANT BEHAVIOR
    ------------------
    - No error handling is performed. It is assumed that the LLM provides valid input.
    - Missing features, missing values, or invalid node types will result in runtime errors.
    - This is intentional: the LLM is expected to construct correct tool inputs.

    EXAMPLES
    --------
    1) Select all Genes:
        resolve_nodes(node_type="Gene", selection_type="all")

    2) Select genes by feature 'name':
        resolve_nodes(
            node_type="Gene",
            selection_type="feature_list",
            feature_names=["name"],
            values={"name": ["DPP4", "GNAS", "GIPR"]}
        )

    3) Multi-feature filtering:
        resolve_nodes(
            node_type="Gene",
            selection_type="feature_list",
            feature_names=["name", "chromosome"],
            values={
                "name": ["DPP4", "GNAS"],
                "chromosome": ["17"]
            }
        )
    """
    df = GRAPH["nodes"][node_type]

    if selection_type == "all":
        result = df
    else:
        result = df
        for feature in feature_names:
            result = result[result[feature].isin(values[feature])]

    return result.to_dict(orient="records")

@tool
def find_paths(
    path_node_types: List[str],
    rel_types_per_step: Optional[List[Optional[List[str]]]] = None,
    start_nodes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Find all paths in the global in-memory graph (GRAPH) that match a requested
    sequence of node types, and return the result as a single tabular dataset
    (one row = one complete path).

    This tool is typically called after selecting starting nodes via `resolve_nodes`.
    It performs step-by-step traversal over edge tables using pandas merges.

    IMPORTANT UPDATE (DIRECTION HANDLING)
    -------------------------------------
    For each step Type(i) -> Type(i+1), this tool uses edges in BOTH directions:
    - Forward edges:  src_type == Type(i)   AND dst_type == Type(i+1)
    - Reverse edges:  src_type == Type(i+1) AND dst_type == Type(i)
      Reverse edges are inverted (src_id <-> dst_id) so they behave as Type(i) -> Type(i+1).

    This is required because some relations in the provided dataset are stored in the
    opposite direction of the user-requested path (e.g., Anatomy -> Gene while the
    user requests Gene -> Anatomy).

    INPUTS
    ------
    path_node_types : list[str]
        Ordered list of node types describing the path shape.
        Example:
            ["Gene", "CellType"]
            ["Gene", "Anatomy", "Anatomy"]

    rel_types_per_step : list[list[str] | null] | null
        Optional relation constraints per step. Length must be len(path_node_types) - 1.

        - If rel_types_per_step is null:
            Use ALL relations available in GRAPH["edges"] that connect the two types
            (considering both forward and reverse directions, with reverse inverted).

        - If rel_types_per_step is provided:
            For each step i, rel_types_per_step[i] can be:
              * null       -> use ALL relations for that step (both directions)
              * list[str]  -> use ONLY those rel_type values for that step (both directions)

        Example for ["Gene","Anatomy","Anatomy"] (2 steps):
            rel_types_per_step = [
                ["EXPRESSES_AeG", "UPREGULATES_AuG"],  # step 0 (Gene <-> Anatomy)
                null                                    # step 1 (Anatomy <-> Anatomy)
            ]

    start_nodes : list[dict] | null
        Optional starting nodes for the first node type in path_node_types.
        Usually the output of `resolve_nodes` (list of node rows).
        If null, ALL nodes of the first node type are used.

        Note: only "node_id" is required in each dict; extra keys are ignored.

    OUTPUT
    ------
    list[dict]
        JSON-serializable table where each dict represents one full path.

        The output includes:
        - node_{j}_id columns for each node position j in the path
        - rel_{i}_type columns for each step i (the relation type used)
        - ALL node attributes for each node position, prefixed as:
            node_0_<node_columns>
            node_1_<node_columns>
            ...
          (No node columns are dropped.)

    ASSUMPTIONS
    -----------
    - No validation / error handling is performed.
    - GRAPH is initialized and contains:
        GRAPH["nodes"][node_type] dataframes with at least column "node_id"
        GRAPH["edges"][rel_type] dataframes with at least:
            src_id, src_type, dst_id, dst_type, rel_type
    """

    # ---- 1) Initial paths (node_0 ids) ----
    first_type = path_node_types[0]

    if start_nodes is None:
        paths = GRAPH["nodes"][first_type][["node_id"]].copy()
    else:
        paths = pd.DataFrame(start_nodes)[["node_id"]].copy()

    paths = paths.rename(columns={"node_id": "node_0_id"})

    # ---- 2) Step-by-step traversal ----
    num_steps = len(path_node_types) - 1

    for i in range(num_steps):
        src_type = path_node_types[i]
        dst_type = path_node_types[i + 1]

        step_edges_list = []

        for _, edf in GRAPH["edges"].items():
            # FORWARD edges: src_type -> dst_type
            fwd = edf[(edf["src_type"] == src_type) & (edf["dst_type"] == dst_type)]
            fwd = fwd[["src_id", "dst_id", "rel_type"]]

            # REVERSE edges: dst_type -> src_type, invert them to behave like src_type -> dst_type
            rev = edf[(edf["src_type"] == dst_type) & (edf["dst_type"] == src_type)]
            rev = rev[["src_id", "dst_id", "rel_type"]].rename(columns={"src_id": "dst_id", "dst_id": "src_id"})

            # Optional per-step relation filter
            if rel_types_per_step is not None:
                rels_i = rel_types_per_step[i]
                if rels_i is not None:
                    fwd = fwd[fwd["rel_type"].isin(rels_i)]
                    rev = rev[rev["rel_type"].isin(rels_i)]

            step_edges_list.append(fwd)
            step_edges_list.append(rev)

        step_edges = pd.concat(step_edges_list, ignore_index=True)

        # Rename for merge
        step_edges = step_edges.rename(
            columns={
                "src_id": f"node_{i}_id",
                "dst_id": f"node_{i + 1}_id",
                "rel_type": f"rel_{i}_type",
            }
        )

        # Expand paths
        paths = paths.merge(step_edges, on=f"node_{i}_id", how="inner")

    # ---- 3) Enrich with ALL node columns per position (clean version) ----
    for j, ntype in enumerate(path_node_types):
        ndf = GRAPH["nodes"][ntype].copy()

        # Rename join key to match path column
        ndf = ndf.rename(columns={"node_id": f"node_{j}_id"})

        # Prefix all OTHER columns
        rename_map = {}
        for col in ndf.columns:
            if col != f"node_{j}_id":
                rename_map[col] = f"node_{j}_{col}"
        ndf = ndf.rename(columns=rename_map)

        # Join
        paths = paths.merge(ndf, on=f"node_{j}_id", how="left")

    return paths.to_dict(orient="records")

@tool
def save_csv(data: List[Dict], output_dir: str = "results") -> str:
    """
    Save a list of dictionaries (typically the output of another tool like resolve_nodes
    or find_paths) into a CSV file.

    PARAMETERS
    ----------
    data : list[dict]
        The tabular data to save. It is assumed that this list represents rows,
        and all keys are column names. No validation is performed.

    output_dir : str, optional
        Directory where the CSV file will be saved. If the directory does not exist,
        it is created automatically.

    BEHAVIOR
    --------
    - Converts the list of dictionaries into a pandas DataFrame.
    - Saves it as a CSV file named with a random UUID.
    - Does NOT perform any checks, validation, or error handling.
    - Assumes that 'data' is well-formed and suitable for DataFrame().

    RETURNS
    -------
    str
        The full path to the saved CSV file.

    EXAMPLE (LLM instruction)
    --------------------------
    "Save the result of the previous tool call as a CSV file."

    EXAMPLE TOOL CALL (generated by LLM)
    ------------------------------------
    {
        "tool": "save_csv",
        "data": [
            {"node_id": 1, "node_type": "Gene", "name": "DPP4"},
            {"node_id": 2, "node_type": "Gene", "name": "GNAS"}
        ],
        "output_dir": "results"
    }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    filename = f"{uuid.uuid4().hex}.csv"
    filepath = Path(output_dir) / filename

    df.to_csv(filepath, index=False)

    return str(filepath)

tools = [resolve_nodes, find_paths, save_csv]

tool_node = ToolNode(tools=tools)

model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model='gpt-4o-mini',
    temperature=0
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    response = model.invoke(
        [system_message] + list(state["messages"])
    )

    return {
        "messages": [response]
    }

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if getattr(last_message, "tool_calls", None) else "end"

graph = StateGraph(AgentState)
graph.add_node("init_graph", init_graph_node)
graph.add_node("our_agent", model_call)
graph.add_node("tools", tool_node)

graph.add_edge(START, "init_graph")
graph.add_edge("init_graph", "our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

user_query = """
Find all paths Gene -> Anatomy -> Anatomy.
For Gene nodes, use nodes where name is in [DPP4, GCG, GIPR, GLP1R, GNAI2, GNAS, MME, MMP12, PCSK1, PCSK2, C1QTNF1, GIP].
Use all edge types.
Save result to CSV.
"""

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", user_query)]}
print_stream(app.stream(inputs, stream_mode="values"))







