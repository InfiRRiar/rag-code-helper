up:
	docker compose up --force-recreate --detach --build

down:
	docker compose down

ollama-pull:
	docker exec -it ollama-server ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
