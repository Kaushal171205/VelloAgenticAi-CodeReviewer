from graph.workflow import build_review_graph
app = build_review_graph()
print(app.get_graph().draw_mermaid())
