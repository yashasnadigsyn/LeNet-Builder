import streamlit as st

st.set_page_config(page_title="LeNet Builder: How To")

st.header("How to Use the LeNet Builder Tool?")
st.write("""
Add Layers:
- Expand the Layers section in the sidebar.
- Choose a layer type (Convolution, Pooling, Flatten, Dense).
- Configure parameters (e.g., filters, kernel size, units).
- Click Add Layer.

Add Activation Functions:
- Expand the Activation Functions section.
- Choose an activation (Sigmoid, Tanh, ReLU).
- Click Add Activation.

Connect Nodes:
- Drag from the output handle (right) of one node to the input handle (left) of another.

Delete Connections:
- Right-click an edge or select it and click Delete Selected Edge.

View PyTorch Code:
- The generated code appears below the flow diagram. Copy and use it in your projects.

Example:
""")
st.image("example.png")
