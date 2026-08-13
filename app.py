import streamlit as st

st.title("My first app")
st.write("netizen")
st.header("netibruzz")

with st.sidebar:
    st.header("Settings")

    name = st.text_input("Enter your name:")
    colorPreset = st.selectbox("Choose your color preset",
    ["Light", "Dark", "Red", "GreyScale", "Aquamarine"])
    creativity = st.slider("Choose your creativity", 0.0, 1.0, 0.5)

    if st.button("Save Settings"):
        st.write(f"Saved Settings, your name is {name}, your color preset is {colorPreset}, and your creativity is {creativity}")
