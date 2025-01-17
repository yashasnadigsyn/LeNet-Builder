import streamlit as st

# Set page config
st.set_page_config(page_title="Train Model")

# Title and description
st.title("Train Your LeNet Model")
st.write("""
In this section, you can view the complete PyTorch code for the neural network you built.
Click the **Train** button to celebrate your hard work!
""")

# Original LeNet code
original_lenet_code = """
import torch
import torch.nn as nn

class LeNet5Original(nn.Module):
    def __init__(self):
        super(LeNet5Original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh3(x)
        x = self.fc2(x)
        x = self.tanh4(x)
        x = self.fc3(x)
        return x
"""

# Button to use the original LeNet code
if st.button("Use Original LeNet"):
    st.session_state.pytorch_code = original_lenet_code  # Store the original LeNet code in session state
    st.success("Original LeNet code loaded successfully!")

# Check if the PyTorch code is available in the session state
if 'pytorch_code' not in st.session_state:
    st.error("No neural network has been built yet. Please go back to the **LeNet Builder** page and build your model.")
else:
    # Display the full PyTorch code
    st.subheader("Complete PyTorch Code")
    st.code(st.session_state.pytorch_code, language='python')

    # Train button
    if st.button("Train"):
        st.balloons()  # Celebrate!
        st.success("Training complete! (Just kidding, we didn't actually train anything. I will add this feature in the future.)")
