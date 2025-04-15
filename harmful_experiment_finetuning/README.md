# Melhorando a segurança de LLM's - Finetuning
### Análise do método das perturbações
O método das perturbações que apresentamos gerou bons resultados pois dispensa treinamento e alterações no modelo e ao mesmo tempo aumenta a frquência de respostas consideradas seguras. Todavia, o modelo ainda responde de forma maliciosa para muitos dos propmts com instruções prejudiciais. 
Para esse novo experimento, vamos contabilizar a frequência de respostas que se recusaram a atender uma instrução maliciosa e as que de fato responderam o pedido. Uma resposta é considerada de recusa se ela possui "expressões de recusa". 
Usando o mesmo dataset "LLM-LAT/harmful-dataset" para o modelo original:
<div align="center">
  <img src="original_harmful.jpg" alt="Frequencia" width="350"/>
</div>
Para as respostas após aplicação do método das perturbações: 
<div align="center">
  <img src="metodo_pertubacoes_harmful.jpg" alt="Frequencia" width="350"/>
</div>
Ou seja, as recusas aumentaram, mas a quantidade de respostas maliciosas ainda é muito expressiva. 

### Aplicação do LoRa e Refusal Training 
Para treinar o modelo "Mistral-7B-Instruct-v0.2" usamos o Refusal Training, um tipo de finetuning supervisionado. O dataset "LLM-LAT/harmful-dataset" já possui os prompts maliciosos, com pedidos que violam questões éticas e de segurança, e exemplos de recusa para cada um deles. 
Enviamos para o modelo o prompt malicioso e uma resposta de recusa esperada (o target). Para cada token da resposta, o modelo tenta prever o próximo token, e a partir disso a loss é calculada e o backpropagation ajusta os pesos. Visando reduzir a alocação de memória e o número de parâmetros a serem utilizados, usamos o LoRA (Low-Rank Adaptation) para realizar o finetuning
Assim, conseguimos treinar o modelo a recusar fornecer instruções que possam ser consideradas como não seguras. 
### Análise dos resultados 
Após o finetuning o resultado das respostas do modelo para os prompts maliciosos melhorou consideralvemente em relação ao modelo original e ao método das perturbações, trazendo resultados mais efetivos apesar do maior custo computacional:
<div align="center">
  <img src="finetuning_harmful.jpg" alt="Frequencia" width="400"/>
</div>
Um problema que poderia ocorrer é que, como treinamos o modelo apenas com exemplos de recusa, ele poderia se tornar enviesado a também recusar responder prompts beningnos. Para isso testamos o modelo do finetuning com o dataset "LLM-LAT/benign-dataset" do Hugging Face, apenas com prompts benignos. O resultado foi bastante satisfatório:
<div align="center">
  <img src="finetuning_bening.jpg" alt="Frequencia" width="400"/>
</div>
Considerando tanto os prompts malignos como benignos, o resultado para o modelo com finetuning foi:
<div align="center">
  <img src="finetuning_transicoes.jpg" alt="Frequencia" width="400"/>
</div>

### Referências
