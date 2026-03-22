import json

filepath = r'c:\Users\ivanu\recog_system\gemini_vision.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

CORRECT_IMAGE_DIR = r"C:\Users\ivanu\Desktop\recog_system\imagens"

new_cells = []

# --- CELULA 0: TITULO ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Gemini API: Explorando Visão Computacional\n",
        "\n",
        "Este notebook demonstra como enviar imagens para o modelo Gemini usando a SDK `google-generativeai`.\n",
        "\n",
        "### Configuração\n",
        "Estamos usando `python-dotenv` para gerenciar a chave de API e `typing_extensions` para o output estruturado."
    ]
})

# --- CELULA 1: IMPORTAÇÕES ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import google.generativeai as genai\n",
        "from PIL import Image\n",
        "import matplotlib.pyplot as plt\n",
        "from dotenv import load_dotenv\n",
        "from pathlib import Path\n",
        "import typing_extensions as typing\n",
        "import json\n",
        "\n",
        "load_dotenv()\n",
        "api_key = os.getenv(\"GOOGLE_API_KEY\")\n",
        "if not api_key:\n",
        "    print(\"[ERRO] GOOGLE_API_KEY não encontrada\")\n",
        "else:\n",
        "    genai.configure(api_key=api_key)\n",
        "    print(\"[OK] Conectado ao Google Generative AI\")"
    ]
})

# --- CELULA 2: MARKDOWN LOCALIZAÇÃO ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Localizando as Imagens\n",
        "\n",
        "Definimos o caminho para a pasta de imagens e listamos os arquivos disponíveis para garantir que o notebook encontre os dados."
    ]
})

# --- CELULA 3: CODE LOCALIZAÇÃO ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        f"IMAGE_DIR = Path(r\"{CORRECT_IMAGE_DIR}\")\n",
        "\n",
        "if IMAGE_DIR.exists():\n",
        "    images = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in [\".jpg\", \".png\", \".jpeg\"]])\n",
        "    print(f\"[INFO] Encontradas {{len(images)}} imagens em: {{IMAGE_DIR}}\")\n",
        "    for i, img_path in enumerate(images):\n",
        "        print(f\"{{i}}: {{img_path.name}}\")\n",
        "else:\n",
        "    print(f\"[ERRO] Diretorio {{IMAGE_DIR}} nao encontrado.\")"
    ]
})

# --- CELULA 4: MARKDOWN VISUALIZAÇÃO ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Seleção e Visualização\n",
        "\n",
        "Escolhemos uma imagem da lista (pelo índice) para carregar e visualizar na tela antes de enviar para a IA."
    ]
})

# --- CELULA 5: CODE VISUALIZAÇÃO ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "indice = 3  # Ajuste para a imagem desejada\n",
        "\n",
        "if 'images' in locals() and indice < len(images):\n",
        "    img_path = images[indice]\n",
        "    img = Image.open(img_path)\n",
        "    \n",
        "    plt.figure(figsize=(10, 6))\n",
        "    plt.imshow(img)\n",
        "    plt.title(f\"Visualizando: {img_path.name}\")\n",
        "    plt.axis(\"off\")\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"[ERRO] Indice invalido ou imagens nao carregadas.\")"
    ]
})

# --- CELULA 6: MARKDOWN ANALISE ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Enviando para o Gemini (Output Estruturado)\n",
        "\n",
        "Agora enviamos a imagem para a API com um `response_schema` definido. Isso obriga o modelo a retornar um JSON estruturado."
    ]
})

# --- CELULA 7: CODE ANALISE ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "if 'img' in locals():\n",
        "    class AnaliseImagem(typing.TypedDict):\n",
        "        categoria: str\n",
        "        objetos_detectados: list[str]\n",
        "        descricao_detalhada: str\n",
        "        confianca: float\n",
        "\n",
        "    model = genai.GenerativeModel('gemini-1.5-flash')\n",
        "    print(f\"[INFO] Analisando '{{img_path.name}}'...\")\n",
        "    \n",
        "    prompt = \"Analise a imagem e retorne um JSON estruturado.\"\n",
        "    \n",
        "    response = model.generate_content(\n",
        "        [prompt, img],\n",
        "        generation_config=genai.GenerationConfig(\n",
        "            response_mime_type=\"application/json\",\n",
        "            response_schema=AnaliseImagem\n",
        "        )\n",
        "    )\n",
        "    \n",
        "    resultado = json.loads(response.text)\n",
        "    print(\"\\n--- Resultado Estruturado (JSON) ---\")\n",
        "    print(json.dumps(resultado, indent=2, ensure_ascii=False))\n",
        "else:\n",
        "    print(\"[ERRO] Nenhuma imagem selecionada.\")"
    ]
})

nb['cells'] = new_cells

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook atualizado com 8 células (Markdown + Code).")
