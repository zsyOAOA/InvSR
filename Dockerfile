FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Create workdir
WORKDIR /invsr

# Copy from the files to the container
COPY . /invsr

# Install requeriments
RUN apt update -y && apt install -y ffmpeg libsm6 libxext6

RUN pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install -e ".[torch]"
RUN pip install -r requirements.txt

# Clean up all cached files
RUN pip cache purge && apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/{apt,dpkg,cache,log}/

# Expose gradio port
EXPOSE 7860

# Set listen for 0.0.0.0
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set python as entrypoint pointing to app.py to run the interface by default
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
