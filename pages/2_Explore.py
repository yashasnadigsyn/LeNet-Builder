import streamlit as st
from streamlit_drawable_canvas import st_canvas
from predict import main

# Set page config
st.set_page_config(page_title="LeNet Builder: Explore")

with st.sidebar:
    st.header("Select a Model")
    if st.button("LeNet-5"):
        st.session_state.selected_model = "LeNet-5"

st.header("Test the Models")
st.write("""
This page allows you to test various prebuilt models.
You can select a model from the sidebar to start testing.

**How to use this page:**
1. Select a model from the sidebar.
2. The selected model will be loaded, and you can proceed to test it.
3. Draw a single digit (0–9) on the canvas below to test the model.

Currently, the available model is:
- **LeNet-5**: A classic convolutional neural network designed for handwritten digit recognition.
""")

if 'selected_model' in st.session_state:
    st.write(f"**Selected Model:** {st.session_state.selected_model}")

    st.write("### Draw a Digit (0–9)")
    st.write("Use the canvas below to draw a single digit for testing the selected model.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=8,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        display_toolbar=True,
        key="canvas",
    )

    if canvas_result.image_data is not None:
        st.write("### Prediction")
        prediction = main(canvas_result.image_data)
        st.write(f"**Predicted Digit:** {prediction}")

else:
    st.write("No model selected yet. Please select a model from the sidebar.")
