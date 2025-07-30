FROM python:3.9

WORKDIR /app

# Copy your app code and dependencies
COPY app.py .
COPY requirements.txt .

# Upgrade pip and install dependencies including streamlit
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Make port 8501 available to the host
EXPOSE 8501

# Use exec form of CMD to run streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

