# has bug in drawing edges,not handles the empty pos dict and graph color grades.
# def visualize_graph(G: nx.DiGraph, layout="dot", node_shape="ellipse", node_color="#ADD8E6", edge_color="#808080", font_color="black"):
#     """
#     Visualizes the NetworkX graph using Matplotlib (with Graphviz layout)
#     and displays it in Streamlit, applying styling options.
#     """
#     if not G.nodes() or len(G.nodes())==0:
#
#         st.warning("No nodes to display in the graph.")
#         return
#
#     # Use pydot for layout as it provides Graphviz layout algorithms
#     try:
#         # Set node and edge attributes for drawing
#         # For simplicity, apply global styles. For per-node/edge styles,
#         # you'd set attributes on individual nodes/edges.
#         node_styles = {node: {"shape": node_shape, "fillcolor": node_color, "style": "filled"} for node in G.nodes()}
#         edge_styles = {edge: {"color": edge_color} for edge in G.edges()}
#
#         nx.set_node_attributes(G, node_styles)
#         nx.set_edge_attributes(G, edge_styles)
#
#         # Matplotlib figure setup
#         fig, ax = plt.subplots(figsize=(10, 8))
#
#         # Use pydot_layout for Graphviz layouts
#         pos = nx.drawing.nx_pydot.pydot_layout(G, prog  =layout)
#
#         # Draw nodes
#         if(len(pos)>0 and G.nodes()):
#             nx.draw_networkx_nodes(G, pos,
#                                    node_shape=[G.nodes[node].get("shape", node_shape) for node in G.nodes()],
#                                    node_color=[G.nodes[node].get("fillcolor", node_color) for node in G.nodes()],
#                                    node_size=3000, # Adjust for label readability
#                                    ax=ax)
#
#             # Draw edges
#             nx.draw_networkx_edges(G, pos,
#                                    arrowstyle='->',
#                                    arrowsize=20,
#                                    edge_color=[G.edges[edge].get("color", edge_color) for edge in G.edges()],
#                                    ax=ax)
#
#             # Draw labels
#             nx.draw_networkx_labels(G, pos, font_size=10, font_color=font_color, ax=ax)
#
#             ax.set_title("Generated Graph")
#             ax.set_axis_off() # Hide axes
#             st.pyplot(fig) # Display the matplotlib plot in Streamlit
#
#             # Provide a download button for the generated image
#             buf = io.BytesIO()
#             fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
#             st.download_button(
#                 label="Download Graph as PNG",
#                 data=buf.getvalue(),
#                 file_name="generated_graph.png",
#                 mime="image/png"
#             )
#             plt.close(fig) # Close the figure to free up memory
#
#     except FileNotFoundError:
#         st.error("Error: Graphviz 'dot' executable not found. "
#                  "Please ensure Graphviz is installed and its 'bin' directory is in your system's PATH.")
#         st.info("Installation instructions: https://graphviz.org/download/")
#     except ImportError:
#         st.error("Error: 'pydot' or 'graphviz' Python package not found. "
#                  "Please install them: `pip install pydot graphviz`")
#     except Exception as e:
#         st.error(f"An unexpected error occurred during graph visualization: {e}")
#         st.exception(e) # Show full traceback for debugging

# Debuged one
import streamlit as st
import networkx as nx
import google.generativeai as ai
import json
import matplotlib.pyplot as plt
import io
import os
if "g_api_key" not in os.environ:
    st.error("Error: Gemini API key (g_api_key) environment variable not set. "
             "Please set it to your Gemini API key.")
    st.stop()  # Stop the app
try:
    ai.configure(api_key=os.environ["g_api_key"])
except Exception as e:
    st.error(f"Error occured during connecting to the LLM : {e}")
# interaction with llm
def extract_graph_data_from_prompt(user_prompt: str) -> dict:
    """
    using Gemini LLM to extract nodes and edges from a user prompt.
    """
    model = ai.GenerativeModel(
        "gemini-1.5-flash")
    system_ins = (
        "You are a graph data extractor. Your task is to analyze user prompts "
        "describing relationships between entities and output a JSON object "
        "containing two keys: 'nodes' (a list of unique entity names) "
        "and 'edges' (a list of dictionaries, each with 'source' and 'target' keys). "
        "Infer nodes from the entities mentioned in the relationships. "
        "Handle 'A travels to B via C' as two directed edges: A->C and C->B. "
        "Ensure the output is valid JSON and nothing else. "
        "Example prompt: 'A travels to B and B Travels to D via C, C travels to E.'\n"
        "Example output: {\"nodes\": [\"A\", \"B\", \"C\", \"D\", \"E\"], \"edges\": [{\"source\": \"A\", \"target\": \"B\"}, {\"source\": \"B\", \"target\": \"C\"}, {\"source\": \"C\", \"target\": \"D\"}, {\"source\": \"C\", \"target\": \"E\"}]}"
        "If a graph cannot be logically formed from the user prompt (e.g., it's a casual conversation, a non-graph request), "
        "then respond ONLY with the exact string 'NO_GRAPH_POSSIBLE'."  
    )
    try:
        response = model.generate_content([
            system_ins,
            f"User prompt: \"{user_prompt}\""
        ])
        graph_data_str = response.text.strip()
        if graph_data_str == 'NO_GRAPH_POSSIBLE':
            st.warning(
                "Graph cannot be formed from the given prompt try another, we encourage your imagination!")
            return {"nodes": [], "edges": []}

        # extracting json(if gemini gives response with code hyphens)
        if graph_data_str.startswith("```json"):
            graph_data_str = graph_data_str[7:]  # Remove ```json
        if graph_data_str.endswith("```"):
            graph_data_str = graph_data_str[:-3]  # Remove ```
        graph_data_str = graph_data_str.strip() 
        # parsing json string
        graph = json.loads(graph_data_str)
        # isinstance() returns the boolean output by checking whether the passed obj is the passed class or not.
        if not isinstance(graph, dict) or "nodes" not in graph or "edges" not in graph:
            raise ValueError("LLM did not return the expected JSON structure (missing 'nodes' or 'edges').")
        if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
            raise ValueError("LLM cannot process the prompt properly cannot generate graph this time, try another time!")
        return graph
    except json.JSONDecodeError as e:
        st.error(f"Error: LLM returned invalid JSON. Please refine your prompt or try again. Details: {e}")
        st.code(f"LLM Output (potential error source):\n{graph_data_str}", language="json")
        return {"nodes": [], "edges": []}
    except Exception as e:
        st.error(f"An error occurred during Gemini API call: {e}")
        st.exception(e)
        return {"nodes": [], "edges": []}

# --- Graph Building Function ---
def build_graph_from_data(graph: dict) -> nx.DiGraph:
    # building graph using networkx lib
    G = nx.DiGraph()  # Directed graph (A->B)
    for node in graph.get("nodes", []):
        G.add_node(node)
    # Add edges
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            G.add_edge(source, target)
        else:
            st.warning(f"Skipping malformed edge: {edge}")
    return G


# --- Graph Visualization Function ---
def visualize_graph(G: nx.DiGraph, layout="dot", node_shape="o", node_color="#ADD8E6", edge_color="#808080",
                    font_color="black",arrwstyle='-|>',arrw_size=60):
    """
    Visualizing the networkX graph using matplotlib (with graphviz layout)
    and displays it in streamlit, applying styling options.
    """
    if not G.nodes():
        st.warning("The extracted data did not form a graph with any nodes. Please check your prompt.")
        return
    # creating matplotlib figure.
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        # pydot layout
        try:
            pos = nx.drawing.nx_pydot.pydot_layout(G, prog=layout)
        except:
            # using spring layout if first one fails
            st.warning("Using spring layout.")
            pos = nx.spring_layout(G, k=2, iterations=50)
        #     if no nodes are processed
        if not pos or len(pos) == 0:
            st.warning("Layout algorithm couldn't position any nodes. Using spring layout as fallback.")
            pos = nx.spring_layout(G, k=2, iterations=50)
        nodes_with_pos = [node for node in G.nodes() if node in pos]
        if not nodes_with_pos:
            st.error(
                "Critical error: No nodes could be positioned for drawing. This shouldn't happen with spring layout.")
            plt.close(fig)
            return
        st.write(f"Graph Data: Graph has -> {len(G.nodes())} nodes , no of nodes positioned are {len(nodes_with_pos)}")

        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodes_with_pos,  # only nodes that have positions
                               node_shape=node_shape,
                               node_color=node_color,
                               node_size=2200,
                               ax=ax)
        #checking if the src and dst edges are present
        edges_to_draw = [(u, v) for u, v in G.edges() if u in pos and v in pos]
        if edges_to_draw:
                nx.draw_networkx_edges(G, pos,
                                       edgelist=edges_to_draw,
                                       arrowstyle=arrwstyle,
                                       arrowsize=arrw_size,
                                       edge_color=edge_color,
                                       arrows=True,
                                       ax=ax)

        # Draw labels
        labels_to_draw = {node: node for node in nodes_with_pos}
        nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=10, font_color=font_color, ax=ax)
        ax.set_title("Generated Graph")
        ax.set_axis_off()
        st.pyplot(fig)
        # download btn
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        st.download_button(
            label="Download Graph as PNG",
            data=buf.getvalue(),
            file_name="graph_viz.png",
            mime="image/png"
        )
        plt.close(fig)

    except FileNotFoundError:
        st.error("Error: Graphviz not found. Using spring layout instead.")
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color=node_color,
                edge_color=edge_color, node_size=3000, ax=ax)
        ax.set_title("Generated Graph (Spring Layout)")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Visualization error: {e}")
        plt.close(fig)
        # showing info
        st.write("Graph nodes:", list(G.nodes()))
        st.write("Graph edges:", list(G.edges()))


# --- Streamlit Application Layout ---
def main():
    st.set_page_config(page_title="graph_viz", layout="centered")
    st.title("LLM Powered Graph 📈 Visualizer")
    st.markdown("""
    
    Enter the prompt in natural language in the bellow input box!\n
    Experiment with different styling options in the sidebar.
    """)
    user_prompt = st.text_area(
        "Describe your graph (e.g., 'Project Alpha depends on Project Beta and Project Gamma.'):",
        "A travels to B and B Travels to D via C, C travels to E.",
        height=100,
    )
    st.sidebar.header("Graph Styling")
    layout_option = st.sidebar.selectbox(
        "Graph Layout Algorithm:",
        ["dot", "neato", "fdp", "sfdp", "circo", "twopi"],
        help="Different algorithms arrange nodes differently. 'dot' is hierarchical, 'neato' is spring-based."
    )
    node_shape = st.sidebar.selectbox(
        "Node Shape:",
        ["Circle", "Square", "Triangle", "Diamond"],
        help="Geometric shape of the nodes."
    )
    arrw_style = st.sidebar.selectbox(
        "Edge ArrowStyle:",
        ["Directed","Bi-Directional","Fancy","Square Brackets"],
        help="Pick the desired ArrowStyle that will be in the graph between nodes."
    )
    arrw_size = st.sidebar.slider(
        "Select the Arrow Size",60,500
    )
    node_color_option = st.sidebar.color_picker(
        "Node Fill Color:", "#78BFE0",  #light blue by default
        help="Color of the node's interior."
    )
    edge_color_option = st.sidebar.color_picker(
        "Edge Color:", "#3B3A3A",  # black
        help="Color of the connecting lines (edges)."
    )
    font_color_option = st.sidebar.color_picker(
        "Text Color:", "#000000",  # Black
        help="Color of the node labels."
    )
    st.markdown("---")  # sep
    css = """
    <style>
    .stButton > button,.stDownloadButton > button{
    border:rgba(239, 239, 240, 0.41) solid 1px;
    transition : background 0.4s ease, box-shadow 0.3s ease-out, color 0.3s ease-in-out;
    }
    .stButton > button:hover,.stButton > button:focus,.stDownloadButton > button:hover,.stDownloadButton > button:focus{
        border:rgba(239, 239, 240, 0.41) solid 1px;
        outline:none !important;
        background: linear-gradient(135deg, #454cc6 2%, #8b90f1 98%);
        box-shadow: 5px 8px 12px rgba(148, 152, 235, 0.4);
        color: #ffffff !important;
    }
     h1{
    font-family:  'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    }
    </style>
    """
    st.markdown(css,unsafe_allow_html=True)
    if st.button("Generate Graph 💹"):
        if user_prompt:
            with st.spinner("Processing with LLM"):
                graph = extract_graph_data_from_prompt(user_prompt)
            if graph and graph.get("nodes") is not None and graph.get("edges") is not None:
                graph = build_graph_from_data(graph)
                shape_mapping = {
                    "Circle": "o",
                    "Square": "s",
                    "Triangle": "^",
                    "Diamond": "D"
                }
                matplotlib_shape = shape_mapping.get(node_shape, "o")
                arr_styles = {
                    "Directed":'-|>',
                    "Bi-Directional":'<->',
                    "Fancy":'fancy',
                    "Square Brackets":']-['
                }
                arrw_style = arr_styles.get(arrw_style,'-|>')
                visualize_graph(
                    graph,
                    layout=layout_option,
                    node_shape=matplotlib_shape,
                    node_color=node_color_option,
                    edge_color=edge_color_option,
                    font_color=font_color_option,
                    arrwstyle=arrw_style,
                    arrw_size=arrw_size
                )
            else:
                pass

        else:
            st.warning("Please enter a description to generate a graph.")

    # At the bottom of your app
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 1em;'>
        🐞 Found a bug? <a href='mailto:balasubrahmanyammantri@gmail.com?subject=Bug Report'>Email us</a> 
        or report it below
        </div>
        """,
        unsafe_allow_html=True
    )

    st.text_input("Quick bug report:",
                               placeholder="Briefly describe the issue...",
                               label_visibility="collapsed")
    if(st.button("Report Bug")):
        st.success("Bug Reported, Email us if you want to contact.")
    st.markdown("""
    ---
    *Designed & Developed by $$Balu$$*\n
    *Graph Visualisation powered by Gemini LLM & Networkx.*
    """)
if __name__ == "__main__":
    main()
