import streamlit as st
import time
from DiT import *

path = "./"
CASEY_MODEL = f"{path}/model/dit_cfg_8_LAYERs_32_HEADs_256_EMBD_DIM_400_TMAX_MNIST_epoch_29.pt"

def main():

    # Enable CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiT(N_EMBD,
                N_HEAD,
                N_FF,
                N_LAYER,
                n_class=N_CLASS,
                n_channel=IMG_CHANNELS,
                img_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                t_max=T_MAX,
                beta_min=BETA_MIN,
                beta_max=BETA_MAX,
                schedule=SCHEDULE,
                s=0.008,
                norm_first=True,
                device=device,
                dropout=0)
    model.to(device)

    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')

    model.load_state_dict(torch.load(CASEY_MODEL, map_location=device)["model_state_dict"])
    print("Model loaded")

    if 'frames' not in st.session_state:
        st.session_state.frames = None
    if 'frame_index' not in st.session_state:
        st.session_state.frame_index = 0

    st.title("Diffusion Transformers 🌀💻🔢")

    st.header("What are Diffusion Models?")
    st.write("""
    Diffusion models are a class of generative models that learn to generate data by reversing a diffusion process. 
    The diffusion process repeatedly adds noise to data until it turns into random noise. The model then learns to 
    reverse this process, starting from noise and gradually generating data. They have been particularly successful 
    in generating high-quality images.
    """)

    st.subheader("Let's get cookin!")
    
    col1, col2 = st.columns([0.6, 0.4], gap = "small")
    with col1:
        generate_option = st.radio(
            "Choose some numbers to generate:", ["**Specific Digit**", ":rainbow[ALL THE NUMBERS]"],
            captions = ["Choose from 0 through 9.", "All of them."], horizontal=True)
    with col2:
        if generate_option == "**Specific Digit**":
            label = st.number_input("Choose a number from 0 to 9:", min_value=0, max_value=9, value=0, step=1)
        else:
            label = None
    
    with st.expander("Diffusion Parameters (advanced settings 🤠)", expanded=False):
        st.write("Set parameters for the denoising process. Mouse over the '?' symbol to learn about each parameter.")
        st.markdown("##### Output batch size (Number of rows to generate)")
        num_rows = st.number_input("Choose the number of images to generate (10 * number of rows)",
                                    min_value=1, max_value=8, value=3, step=1,
                                    help="Larger values will take longer to process.")
        st.markdown("##### Time steps")
        time_steps = st.number_input("Set the number of diffusion time steps",
                                    min_value=10, max_value=T_MAX, value=20, step=20,
                                    help="Controls the detail of the diffusion process. \
                                        More steps can lead to finer details, but small values like 20 suffice. \
                                        Each time step results in the model predicting the noise from the image 1 more time.")
        st.markdown("##### Guidance scale")
        guidance_scale = st.number_input("Adjust the guidance scale",
                                        min_value=0, max_value=50, value=5, step=1,
                                        help="Influences the adherence to the class label. \
                                            Higher values enforce more label consistency. \
                                            Try seeing what happens when the guidance scale is zero!")
        st.markdown("##### Negative label")
        negative_label_options = list(range(10)) + ["unconditional"]
        negative_label = st.selectbox("Select the negative label condition",
                                    options=negative_label_options, index=len(negative_label_options)-1,
                                    help="Chooses the negative label for image generation. 'unconditional' ignores label guidance. Otherwise, \
                                        the model attempts to generate images of numbers that look UNLIKE the negative label.")

    if st.button("Generate Digits!"):
        with st.spinner("🍳😤 Cooking..."):
            st.session_state.frames = make_denoise_frames(model, label = label, num_rows = num_rows, num_steps = time_steps, scale=guidance_scale, neg_label=negative_label)
    
    frame_placeholder = st.empty()
    if st.session_state.frames is not None:
        animate_button = st.button('Animate the Denoising Process!')
        frame_index = st.slider('Select a specific frame of the denoising process', 0, len(st.session_state.frames)-1)
        st.session_state.frame_index = frame_index
        frame_placeholder.pyplot(st.session_state.frames[frame_index])
        if animate_button:
            for i in range(len(st.session_state.frames)):
                st.session_state.frame_index = i
                frame_placeholder.pyplot(st.session_state.frames[i])
                time.sleep(0.1)  # Add delay for animation effect (100 ms between frames)
                if st.session_state.frame_index != i:
                    break # Stop the loop if the user moves the slider
        if st.checkbox("Compare generated images to the original dataset?"):
            mnist_text = f"({label}'s only)" if label is not None else ""
            st.markdown(f"#### Original MNIST Images {mnist_text}")
            fig = plot_MNIST_images(num_rows, label)
            st.pyplot(fig)

    st.caption('UCLA DSU Project Winter 2024: Daniel Mendelevitch, Terry Ming, Casey Tattersall, Sean Tjoa')
    st.caption('Data Attribution: LeCun, Yann, Corinna Cortes, and C. J. C. Burges. \
               MNIST Handwritten Digit Database. 2010, yann.lecun.com/exdb/mnist/')

if __name__ == "__main__":
    main()