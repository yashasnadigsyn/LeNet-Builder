import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
from uuid import uuid4

# Set page config
st.set_page_config(page_title="LeNet Builder")

def generate_pytorch_code(nodes, edges):
    node_map = {node.id: node for node in nodes}

    start_node = next((node for node in nodes if node.data['content'] == 'Image'), None)
    if not start_node:
        return "No input node (Image) found in the flow."

    code = "import torch\nimport torch.nn as nn\n\n"
    code += "class LeNetBuilder(nn.Module):\n"
    code += "    def __init__(self):\n"
    code += "        super(LeNetBuilder, self).__init__()\n"

    current_node = start_node
    layer_count = 1
    layer_names = {}
    while current_node:
        if current_node.data['content'] == 'Image':
            pass
        elif current_node.data['content'].startswith('Conv'):
            filters = int(current_node.data['content'].split('\n')[1].split(': ')[1])
            kernel_size = int(current_node.data['content'].split('\n')[2].split(': ')[1].split('x')[0])
            stride = int(current_node.data['content'].split('\n')[3].split(': ')[1])
            code += f"        self.conv{layer_count} = nn.Conv2d(in_channels=1, out_channels={filters}, kernel_size={kernel_size}, stride={stride})\n"
            code += f"        self.sigmoid{layer_count} = nn.Sigmoid()\n"
            layer_names[current_node.id] = f"conv{layer_count}"
            layer_count += 1
        elif current_node.data['content'].startswith('Pool'):
            pool_size = int(current_node.data['content'].split('\n')[1].split(': ')[1].split('x')[0])
            strides = int(current_node.data['content'].split('\n')[2].split(': ')[1])
            code += f"        self.pool{layer_count} = nn.MaxPool2d(kernel_size={pool_size}, stride={strides})\n"
            layer_names[current_node.id] = f"pool{layer_count}"
            layer_count += 1
        elif current_node.data['content'] == 'Flatten':
            code += f"        self.flatten{layer_count} = nn.Flatten()\n"
            layer_names[current_node.id] = f"flatten{layer_count}"
            layer_count += 1
        elif current_node.data['content'].startswith('Dense'):
            units = int(current_node.data['content'].split('\n')[1].split(': ')[1])
            code += f"        self.fc{layer_count} = nn.Linear(in_features=120, out_features={units})\n"
            code += f"        self.sigmoid{layer_count} = nn.Sigmoid()\n"
            layer_names[current_node.id] = f"fc{layer_count}"
            layer_count += 1

        next_edge = next((edge for edge in edges if edge.source == current_node.id), None)
        if next_edge:
            current_node = node_map.get(next_edge.target)
        else:
            current_node = None

    code += "\n    def forward(self, x):\n"
    current_node = start_node
    while current_node:
        if current_node.data['content'] == 'Image':
            code += "        x = x  # Input image\n"
        elif current_node.id in layer_names:
            layer_name = layer_names[current_node.id]
            if layer_name.startswith("conv"):
                code += f"        x = self.{layer_name}(x)\n"
                code += f"        x = self.sigmoid{layer_name[4:]}(x)\n"
            elif layer_name.startswith("pool"):
                code += f"        x = self.{layer_name}(x)\n"
            elif layer_name.startswith("flatten"):
                code += f"        x = self.{layer_name}(x)\n"
            elif layer_name.startswith("fc"):
                code += f"        x = self.{layer_name}(x)\n"
                code += f"        x = self.sigmoid{layer_name[2:]}(x)\n"

        next_edge = next((edge for edge in edges if edge.source == current_node.id), None)
        if next_edge:
            current_node = node_map.get(next_edge.target)
        else:
            current_node = None

    code += "        return x\n"
    return code

if 'curr_state' not in st.session_state:
    nodes = [
        StreamlitFlowNode("Image", (0, 0), {'content': 'Image'}, node_type='input', source_position='right', draggable=True, style={'backgroundColor': '#FFCCCB'}),
        StreamlitFlowNode("Softmax", (40, 0), {'content': 'Softmax'}, 'output', target_position='left', draggable=True, style={'backgroundColor': '#ADD8E6'})  # Moved to x=40
    ]
    edges = []
    st.session_state.curr_state = StreamlitFlowState(nodes, edges)

with st.sidebar:
    st.header("LeNet Builder")

    with st.expander("Layers"):
        layer_type = st.selectbox("Select Layer Type", ["Convolution", "Pooling", "Flatten", "Dense"])

        if layer_type == "Convolution":
            filters = st.number_input("Filters", min_value=1, value=6)
            kernel_size = st.number_input("Kernel Size", min_value=1, value=5)
            stride = st.number_input("Stride", min_value=1, value=1)
        elif layer_type == "Pooling":
            pool_size = st.number_input("Pool Size", min_value=1, value=2)
            strides = st.number_input("Strides", min_value=1, value=2)
        elif layer_type == "Dense":
            units = st.number_input("Units", min_value=1, value=120)

        if st.button("Add Layer"):
            new_node_id = str(uuid4())

            if layer_type == "Convolution":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': f'Conv\nFilters: {filters}\nKernel: {kernel_size}x{kernel_size}\nStride: {stride}'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#90EE90'},
                    resizing=True
                )
            elif layer_type == "Pooling":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': f'Pool\nSize: {pool_size}x{pool_size}\nStrides: {strides}'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#FFD700'}
                )
            elif layer_type == "Flatten":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': 'Flatten'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#DDA0DD'}
                )
            elif layer_type == "Dense":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': f'Dense\nUnits: {units}'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#87CEEB'}
                )

            st.session_state.curr_state.nodes.append(new_node)
            st.rerun()

    with st.expander("Activation Functions"):
        activation_type = st.selectbox("Select Activation Function", ["Sigmoid", "Tanh", "ReLU"])

        if st.button("Add Activation"):
            new_node_id = str(uuid4())

            if activation_type == "Sigmoid":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': 'Sigmoid'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#FFA07A'}
                )
            elif activation_type == "Tanh":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': 'Tanh'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#FF7F50'}
                )
            elif activation_type == "ReLU":
                new_node = StreamlitFlowNode(
                    new_node_id, (1, 0),
                    {'content': 'ReLU'},
                    'default', 'right', 'left',
                    style={'backgroundColor': '#6495ED'}
                )

            st.session_state.curr_state.nodes.append(new_node)
            st.rerun()

    if st.button("Delete Selected Edge"):
        if st.session_state.curr_state.selected_id:
            selected_edge = next((edge for edge in st.session_state.curr_state.edges if edge.id == st.session_state.curr_state.selected_id), None)
            if selected_edge:
                st.session_state.curr_state.edges = [edge for edge in st.session_state.curr_state.edges if edge.id != selected_edge.id]
                st.session_state.curr_state.selected_id = None  # Clear the selected ID
                st.rerun()
            else:
                st.warning("No edge selected. Please select an edge to delete.")
        else:
            st.warning("No edge selected. Please select an edge to delete.")

# Main content
st.title("Build a LeNet!")
st.header("What is LeNet?")
st.write("""
LeNet is one of the earliest convolutional neural networks (CNNs), designed by Yann LeCun and his colleagues in the 1990s.
It was primarily used for handwritten digit recognition and became the foundation for modern deep learning architectures.

**Fun Fact:** To this day, some ATMs still run the code that Yann LeCun and his colleague Leon Bottou wrote in the 1990s!
""")

# Display the flow diagram
st.session_state.curr_state = streamlit_flow(
    'lenet_builder_flow',
    st.session_state.curr_state,
    layout=TreeLayout(direction='down', node_node_spacing=100),
    fit_view=True,
    height=500,
    animate_new_edges=True,
    enable_node_menu=True,
    enable_edge_menu=True,
    enable_pane_menu=True,
    get_edge_on_click=True,
    get_node_on_click=True,
    show_minimap=True,
    hide_watermark=True,
    allow_new_edges=True,
    min_zoom=0.1
)

# Display the generated PyTorch code
st.subheader("Generated PyTorch Code")
pytorch_code = generate_pytorch_code(st.session_state.curr_state.nodes, st.session_state.curr_state.edges)
st.session_state.pytorch_code = pytorch_code
st.code(pytorch_code, language='python')
