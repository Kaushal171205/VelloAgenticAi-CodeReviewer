"""
graph/studio_graph.py — Entrypoint for LangGraph Studio.

LangGraph Studio provides its own Postgres-based checkpointer at runtime,
so we compile the graph without enabling human_review (which would try to
create a SQLite checkpointer). The human_node and interrupt logic still
exist in the graph definition and work when invoked via the Streamlit app.
"""

from graph.workflow import build_review_graph

# Compile without human_review to avoid SQLite checkpointer creation.
# Studio will inject its own Postgres checkpointer for state persistence.
graph = build_review_graph(enable_critic=True, enable_human_review=False)
