up:
	docker compose up --build --force-recreate --detach
	make ollama-up

down:
	docker compose down
	make ollama-down

ollama-up:
	docker run -d \
	 	--name ollama-server \
		--gpus=all \
		-v ollama:/root/.ollama \
		-p 11434:11434 \
		ollama/ollama
	docker exec -it ollama-server ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

ollama-down:
	docker stop ollama-server
	docker rm ollama-server