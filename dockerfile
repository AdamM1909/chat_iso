FROM python:3.11.6

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=OPENAI_API_KEY

ENV PORT 8080
ENV HOST 0.0.0.0

CMD ["streamlit", "run", "v2.py", "--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false", "--server.port", "8080"]



