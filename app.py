import os
import streamlit as st
from groq import Groq
from sklearn.metrics import f1_score, precision_score, recall_score

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Example dataset for few-shot prompting
dataset_examples = [
    {
        "prompt": "Generate a YouTube script on the topic: Machine Learning\n\n### Example Script:\n\nMachine learning is a subset of artificial intelligence...",
        "completion": "Machine learning has revolutionized many industries by enabling computers to learn from data..."
    },
    {
        "prompt": "Generate a YouTube script on the topic: Climate Change\n\n### Example Script:\n\nClimate change is a significant and lasting change in the Earth's climate...",
        "completion": "Climate change affects every aspect of our planet, from weather patterns to ecosystems..."
    }
    
]

def create_few_shot_prompt(topic, script_type):
    prompt = f"Generate a {script_type} YouTube script on the topic: {topic}\n\n"
    for example in dataset_examples:
        prompt += example["prompt"] + "\n\n" + example["completion"] + "\n\n"
    prompt += "### Generated Script:"
    return prompt

def generate_script(prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

def evaluate_script(true_script, generated_script):
    true_words = set(true_script.split())
    generated_words = set(generated_script.split())
    true_positives = len(true_words & generated_words)
    false_positives = len(generated_words - true_words)
    false_negatives = len(true_words - generated_words)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, true_positives, false_positives

# Streamlit application
st.title("YouTube Script Generator")
st.write("Generate long-form YouTube scripts and short-form scripts for reels or shorts using the Groq API Llama3 model.")

st.sidebar.header("User Input")
topic = st.sidebar.text_input("Enter a topic:")
script_type = st.sidebar.selectbox("Select script type:", ("short", "long"))

if st.sidebar.button("Generate Script"):
    with st.spinner("Generating script..."):
        few_shot_prompt = create_few_shot_prompt(topic, script_type)
        script = generate_script(few_shot_prompt)
    if script:
        st.subheader("Generated Script")
        st.write(script)

        # Example true script for evaluation (replace with actual true script)
        true_script = dataset_examples[0]["completion"]  # Example true script from dataset

        precision, recall, f1, true_positives, false_positives = evaluate_script(true_script, script)

        st.subheader("Evaluation Metrics")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")
        st.write(f"True Positives: {true_positives}")
        st.write(f"False Positives: {false_positives}")
    else:
        st.error("Failed to generate script.")
