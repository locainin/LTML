# LTML (Long-Term Machine Learning)
A Sleep-Like Function for AI Models

## What is LTML?
LTML is an experimental framework designed to give AI models a “sleep phase” to process, refine, and retain long-term memory—just like a human brain consolidates information during sleep.

# Why?
Current AI models reset context every session, forcing them to constantly relearn the same things like a stateless goldfish. LTML fixes this by:

- ✅ Saving past interactions to a structured database.

- ✅ Running a background memory filter to consolidate important knowledge while the model is idle.

- ✅ Injecting optimized memory back into the model when it wakes up, improving recall and learning efficiency.

# **How Does It Work?**

## **Inference Mode**
  - The AI model interacts normally.
  - All context gets logged into a **memory database** (FAISS, Chroma, SQLite, etc.).

## **Sleep Phase (Model Shuts Down)**
  - **During sleep, a filtering algorithm processes stored memory by** 
    - Sorting **important vs. useless** data.
    - Merging **redundant information**.
    - Extracting **key lessons** to structure memory efficiently.
  - Memory gets **structured and indexed** for efficient retrieval.

## **Wake-Up Phase**
  - On restart, the AI retrieves the **optimized memory** and injects key knowledge back into context.
  - The model now **remembers past interactions** without retraining or fine-tuning.

## **Sleep-Learning for an Ollama Model**
LTML introduces a **three-phase sleep process** for AI:

1️⃣ **Session Ends → Store Context**  
   - All interactions are **dumped into a memory database** for later processing.

2️⃣ **Model Shuts Down → Sleep Phase Begins**  
   - A **filter net** (lightweight autoencoder or rule-based system) processes the stored memory by:
     - **Sorting** important vs. junk data.
     - **Merging** redundant knowledge.
     - **Refining** key concepts for structured memory.

3️⃣ **Wake-Up → Load Optimized Memory**  
   - On restart, the AI retrieves **only relevant and optimized memory**.
   - This allows for **context-aware responses without bloating itself with unnecessary junk**.

💡 **Goal:** Automate this process so AI models can learn, forget unnecessary data, and refine their memory **without requiring fine-tuning**.

---

## **Installation**
### **Requirements**
Ensure you have Python and a virtual environment set up.

```bash
# Clone the repo
git clone https://github.com/locainin/LTML.git
cd LTML

# Activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
