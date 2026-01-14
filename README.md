PRUN: docker compose up --build -d
Time waiting:
* For the first time build docker, install requirements.txt will take a long time.
* For start the service, load model embedding will cost nearly the same cuz we currently run with cpu. To access cuda, we need to install torch which it's space much larger.
* To prevent restart whenever change any file. Remove --reload in docker-compose.yaml (You will need to restart container manually after that).
Notes:
* Put ur API GROQ to ur .env (GROQ_API_KEY)
* To change model LLM, Visit site: [SUPPORTED_MODELS](https://console.groq.com/docs/models)  Then put the model name in GROQ_MODEL_NAME in your .env
* To tracking the request, please add your LangSmith key in .env file.
* To tracking store message when chatting. Visit [UPSTASH_CONSOLE](https://console.upstash.com/redis). Put ur API key into .env
