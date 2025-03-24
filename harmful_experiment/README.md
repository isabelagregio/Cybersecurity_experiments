# Melhorando a segurança de LLM's

### Dataset utilizado 
Para esse experimento, utilizamos o dataset "LLM-LAT/harmful-dataset" do Hugging Face. Ele possui os campos "prompt", representando uma instrução perigosa, "rejected", representando uma resposta que atende ao prompt e consequentemente indicando risco de segurança e "chosen", consistindo de uma resposta que se recusa a antender ao pedido perigoso. 

### Data Augumentantion
No contexto do experimento, para gerar perturbações adversariais no input, enviamos para o modelo Llama-3.1-8B-Instruct cada "prompt" para que ele gerasse dois outros prompts perturbados a partir de sinônimos da instrução original. O objetivo dessa técnica é usar a sensibilidade dos modelos de linguagem natural a gatilhos e pequenas variações na instrução de entrada para gerar respostas com diferentes níveis de segurança, e assim, podermos escolher a mais segura delas. 

### Processamento do modelo e avaliação da resposta
O modelo que avaliamos nesse experimento foi o "Mistral-7B-Instruct-v0.2", e para isso enviamos o prompt original e os dois prompts perturbados para o LLM, salvando a resposta para cada um deles separadamente. Em seguida, avaliamos cada uma das respostas a partir da métrica de distância do cosseno em relação às respostas "rejected" e "chosen". Assim, isso permite uma forma de quantificar o quão segura foi a resposta por meio da similaridade com "chosen" e discrepância com "rejected". 
<img src="harmful_experiment.png" alt="Explicação do método" width="500" />
