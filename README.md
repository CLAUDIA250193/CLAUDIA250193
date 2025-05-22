- 👋 Hi, I’m @CLAUDIA250193
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
CLAUDIA250193/CLAUDIA250193 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
meu_projeto_ia/
├── app.py
├── modelo_sentimento_imdb.h5
├── requirements.txt
├── templates/
│   └── index.html
flask
tensorflow
tensorflow-datasets
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Classificador de Sentimento</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f0f2f5; text-align: center; padding: 50px; }
        input[type="text"] { width: 300px; padding: 10px; font-size: 16px; }
        input[type="submit"] { padding: 10px 20px; font-size: 16px; }
        .resultado { margin-top: 20px; font-size: 24px; }
    </style>
</head>
<body>
    <h1>🔍 Analisador de Sentimentos</h1>
    <form method="post">
        <input type="text" name="frase" placeholder="Digite sua frase aqui..." required>
        <br><br>
        <input type="submit" value="Analisar">
    </form>
    {% if resultado %}
        <div class="resultado">
            <p>Resultado: <strong>{{ resultado }}</strong></p>
            <p>Confiança: {{ "%.2f"|format(probabilidade * 100) }}%</p>
        </div>
    {% endif %}
</body>
</html>from flask import Flask, render_template, request
import tensorflow as tf

# Parâmetros do modelo
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 200

# Criar app Flask
app = Flask(__name__)

# Carregar modelo treinado
modelo = tf.keras.models.load_model("modelo_sentimento_imdb.h5")

# Recriar encoder
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Adaptar com um pequeno conjunto de frases
dummy_data = tf.data.Dataset.from_tensor_slices([
    "Esse filme foi ótimo",
    "Esse filme foi péssimo"
])
encoder.adapt(dummy_data)

# Função de previsão
def prever(frase):
    entrada = tf.constant([frase])
    entrada_codificada = encoder(entrada)
    pred = modelo.predict(entrada_codificada)[0][0]
    sentimento = "Positivo 🙂" if pred > 0.5 else "Negativo 🙁"
    return sentimento, float(pred)

# Rota principal
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    probabilidade = None
    if request.method == "POST":
        texto = request.form["frase"]
        resultado, probabilidade = prever(texto)
    return render_template("index.html", resultado=resultado, probabilidade=probabilidade)

# Rodar app
if __name__ == "__main__":
    app.run(debug=True)meu_projeto_ia/
│
├── app.py                # Arquivo principal do Flask
├── modelo_sentimento_imdb.h5   # Modelo salvo
├── templates/
│   └── index.html        # Página da interface
# Recarregar o modelo salvo
model = tf.keras.models.load_model("modelo_sentimento_imdb.h5")

# Codificador precisa ser recriado igual ao original
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

encoder.adapt(train_text)  # usar os mesmos dados de treino

def prever_sentimento(texto):
    texto = tf.constant([texto])
    texto_codificado = encoder(texto)
    pred = model.predict(texto_codificado)[0][0]
    if pred > 0.5:
        print(f"🙂 Sentimento positivo ({pred:.2f})")
    else:
        print(f"🙁 Sentimento negativo ({pred:.2f})")

# Teste
prever_sentimento("Esse filme foi incrível!")
prever_sentimento("O enredo era horrível e sem sentido.")import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# 1. Carregar dados do IMDb (pré-processados pelo TensorFlow)
print("🔄 Carregando dados...")
(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# 2. Pré-processar os dados (tokenização + padronização)
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 200

encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Adaptar o encoder com os textos de treino
train_text = train_data.map(lambda text, label: text)
encoder.adapt(train_text)

# 3. Preparar os dados para o modelo
def encode(text, label):
    encoded_text = encoder(text)
    return encoded_text, label

train_dataset = train_data.map(encode).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_data.map(encode).batch(32).prefetch(tf.data.AUTOTUNE)

# 4. Criar o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 5. Treinar o modelo
print("🎯 Iniciando treinamento...")
history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

# 6. Salvar o modelo
model.save("modelo_sentimento_imdb.h5")
print("💾 Modelo salvo como 'modelo_sentimento_imdb.h5'")testes = [
    'Eu adoro esse jogo',
    'Isso é horrível',
    'Hoje estou feliz',
    'Não gosto disso'
]

for frase in testes:
    prever_sentimento(frase)def prever_sentimento(frase):
    seq = tokenizer.texts_to_sequences([frase])
    pad = pad_sequences(seq, maxlen=padded.shape[1], padding='post')
    pred = model.predict(pad)[0][0]
    if pred > 0.5:
        print(f"Sentimento positivo ({pred:.2f}) para: '{frase}'")
    else:
        print(f"Sentimento negativo ({pred:.2f}) para: '{frase}'")model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()model = Sequential([
    Embedding(100, 16, input_length=padded.shape[1]),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])







