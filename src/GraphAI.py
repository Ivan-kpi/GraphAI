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
    graph_path: str

def init_graph_node(state: AgentState) -> AgentState:
    """ First node in the StateGraph. Initialize the global GRAPH """
    init_global_graph(state["graph_path"])
    return state

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
    Find all paths in the global in-memory graph that match a requested
    node-type sequence.

    This tool is used after selecting nodes (e.g., via resolve_nodes). It performs
    step-by-step traversal over edge tables and builds a single tabular result where
    each row corresponds to one complete path.

    INPUTS
    ------
    path_node_types : list[str]
        The ordered sequence of node types that defines the path shape.
        Example:
            ["Gene", "CellType"]
            ["Gene", "Anatomy", "Anatomy"]

    rel_types_per_step : list[ list[str] | null ] | null
        Relation constraints per step, with length = len(path_node_types) - 1.

        - If rel_types_per_step is null:
            Use ALL relation types available in GRAPH["edges"] between the two node types
            for each step.

        - If rel_types_per_step is provided:
            For each step i, rel_types_per_step[i] can be:
              * null  -> use ALL relation types for that step
              * list[str] -> use ONLY those rel_type values for that step

        Example for path ["Gene","Anatomy","Anatomy"] (2 steps):
            rel_types_per_step = [
                ["EXPRESSES_AeG", "UPREGULATES_AuG"],  # step 0: Gene->Anatomy
                null                                   # step 1: Anatomy->Anatomy (all)
            ]

    start_nodes : list[dict] | null
        Optional starting node set for the first node type in path_node_types.
        Typically you pass the output of resolve_nodes here (list of node records).
        If null, the tool uses ALL nodes of the first node type.

        Note: only the "node_id" field is required in each dict.

    OUTPUT
    ------
    list[dict]
        JSON-serializable table: one dict = one path (one row).

        Columns include:
        - rel_{i}_type for each step i (the edge rel_type used)
        - all node attributes for each node in the path, prefixed:
            node_0_<original_node_columns>
            node_1_<original_node_columns>
            ...
        Node IDs are stored as:
            node_0_node_id, node_1_node_id, ...

    ASSUMPTIONS
    -----------
    - No validation / error handling is performed.
    - It assumes GRAPH is initialized and contains the requested node/edge tables.
    - It assumes edge tables contain at least:
        src_id, src_type, dst_id, dst_type, rel_type
      and node tables contain node_id (plus any other features).
    """

    first_type = path_node_types[0]

    if start_nodes is None:
        start_df = GRAPH["nodes"][first_type][["node_id"]].copy()
    else:
        start_df = pd.DataFrame(start_nodes)[["node_id"]].copy()

    paths = start_df.rename(columns={"node_id": "node_0_id"})

    # Step-by-step traversal and path expansion
    num_steps = len(path_node_types) - 1

    for i in range(num_steps):
        src_type = path_node_types[i]
        dst_type = path_node_types[i + 1]

        # Build one edge dataframe for this step by scanning all edge tables
        step_edges_list = []
        for _, edf in GRAPH["edges"].items():
            # Filter by direction and types using edge columns
            part = edf[(edf["src_type"] == src_type) & (edf["dst_type"] == dst_type)]

            # Optional relation filter for this step
            if rel_types_per_step is not None:
                rels_i = rel_types_per_step[i]
                if rels_i is not None:
                    part = part[part["rel_type"].isin(rels_i)]

            step_edges_list.append(part[["src_id", "dst_id", "rel_type"]])

        step_edges = pd.concat(step_edges_list, ignore_index=True)

        # Rename for merge
        step_edges = step_edges.rename(
            columns={
                "src_id": f"node_{i}_id",
                "dst_id": f"node_{i+1}_id",
                "rel_type": f"rel_{i}_type",
            }
        )

        # Expand paths by joining on the current node id
        paths = paths.merge(step_edges, on=f"node_{i}_id", how="inner")

    # Enrich with ALL node columns for each position (prefixed)
    for j, ntype in enumerate(path_node_types):
        ndf = GRAPH["nodes"][ntype].copy()

        # Prefix all columns to avoid collisions across different node types/positions
        ndf = ndf.rename(columns={"node_id": f"node_{j}_id"})
        ndf = ndf.add_prefix(f"node_{j}_")

        # But after add_prefix, join key became "node_{j}_node_{j}_id"; fix it:
        # rename "node_0_node_0_id" -> "node_0_id"
        ndf = ndf.rename(columns={f"node_{j}_node_{j}_id": f"node_{j}_id"})

        paths = paths.merge(ndf, on=f"node_{j}_id", how="left")

    # Return as JSON-serializable records
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

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
