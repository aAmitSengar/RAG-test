# RAG System - Detailed Flow Diagram & Architecture

## PHASE 1: SYSTEM INITIALIZATION

### Step 1: Config Initialization
**File:** `src/rag/config.py` → `Config.__init__()`

```
Config.__init__()
│
├── Resolve project directories
│   └── Path(__file__).parent.parent.parent
│       └── Result: project_root = /Users/amit.kumar2/practice/RAG/RAG-test
│
├── Create required directories
│   ├── self.data_dir.mkdir(exist_ok=True)           → data/
│   └── self.models_dir.mkdir(exist_ok=True)         → models/
│
├── Define file paths
│   ├── self.docs_file = data/docs.txt
│   └── self.index_file = data/faiss.index
│
├── Resolve model paths (Priority: env > local > HuggingFace)
│   ├── _resolve_model_path() [Embedding Model]
│   │   ├── Check: env var "EMB_MODEL"
│   │   ├── Check: models/all-MiniLM-L6-v2/ exists locally
│   │   └── Fallback: "all-MiniLM-L6-v2" (HuggingFace ID)
│   │       └── Result: self.emb_model = "all-MiniLM-L6-v2"
│   │
│   ├── _resolve_model_path() [Generation Model]
│   │   ├── Check: env var "GEN_MODEL"
│   │   ├── Check: models/t5-small/ exists locally
│   │   └── Fallback: "t5-small" (HuggingFace ID)
│   │       └── Result: self.gen_model = "t5-small"
│   │
│   ├── @property emb_model_is_local
│   │   └── Returns: bool (is model stored locally?)
│   │
│   └── @property gen_model_is_local
│       └── Returns: bool (is model stored locally?)
│
├── Setup SSL certificates [macOS specific]
│   └── _setup_ssl_certificates()
│       ├── Check: sys.platform == "darwin"
│       ├── Set: REQUESTS_CA_BUNDLE environment variable
│       └── Set: SSL_CERT_FILE environment variable
│
├── Load runtime configuration
│   ├── self.retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))
│   └── self.use_local_only = os.getenv("USE_LOCAL_ONLY", "false")
│
└── Log configuration
    └── logger.info("✓ Configuration ready")
```

---

### Step 2: Retriever Initialization
**File:** `src/rag/retriever.py` → `Retriever.__init__()`

```
Retriever.__init__(config)
│
├── Store config reference
│   └── self.config = config
│
├── Load Documents from file
│   ├── with open(config.docs_file, "r") as f:
│   ├── self.documents = f.read().split("\n---\n")
│   │   └── Result: ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
│   │
│   └── logger.info(f"Loaded {len(documents)} documents")
│
├── Load Embedding Model (SentenceTransformer)
│   ├── if config.emb_model_is_local:
│   │   └── model = SentenceTransformer(config.emb_model)
│   │       └── Load from: models/all-MiniLM-L6-v2/
│   │
│   └── else:
│       └── model = SentenceTransformer("all-MiniLM-L6-v2")
│           ├── Download from HuggingFace Hub (first time only)
│           └── Cache location: ~/.cache/huggingface/
│
├── Create or Load FAISS Index
│   ├── if config.index_file.exists():
│   │   ├── self.index = faiss.read_index(str(config.index_file))
│   │   └── logger.info("Loaded existing FAISS index")
│   │
│   └── else:
│       ├── Encode all documents
│       │   └── embeddings = model.encode(documents)
│       │       ├── Shape: (num_docs, 384)
│       │       └── Example: [[0.123, -0.456, ...], [0.789, ...], ...]
│       │
│       ├── Initialize FAISS IndexFlatL2
│       │   └── self.index = faiss.IndexFlatL2(384)
│       │       └── Dimension = 384 (embedding vector size)
│       │
│       ├── Add embeddings to index
│       │   └── self.index.add(embeddings)
│       │
│       ├── Save index to disk
│       │   └── faiss.write_index(self.index, str(config.index_file))
│       │       └── File size: ~10MB (for 100 docs)
│       │
│       └── logger.info("Created & saved FAISS index")
│
└── Store for later use
    ├── self.model = embedding_model
    └── self.embedding_dim = 384
```

---

### Step 3: Generator Initialization
**File:** `src/rag/generator.py` → `Generator.__init__()`

```
Generator.__init__(config)
│
├── Store config reference
│   └── self.config = config
│
├── Load T5 Tokenizer
│   ├── if config.gen_model_is_local:
│   │   └── tokenizer = T5Tokenizer.from_pretrained(config.gen_model)
│   │       └── Load from: models/t5-small/
│   │
│   └── else:
│       └── tokenizer = T5Tokenizer.from_pretrained("t5-small")
│           ├── Download from HuggingFace (first time only)
│           └── Cache: ~/.cache/huggingface/
│
├── Load T5 Model (PyTorch - Seq2Seq)
│   ├── if config.gen_model_is_local:
│   │   └── model = T5ForConditionalGeneration.from_pretrained(config.gen_model)
│   │       └── Load from: models/t5-small/
│   │
│   └── else:
│       └── model = T5ForConditionalGeneration.from_pretrained("t5-small")
│           ├── Download seq2seq model (~250MB)
│           └── Cache: ~/.cache/huggingface/
│
├── Detect and set device
│   ├── if torch.cuda.is_available():
│   │   └── self.device = "cuda"  (GPU - faster)
│   │
│   └── else:
│       └── self.device = "cpu"   (CPU - slower)
│
├── Move model to device
│   └── model.to(self.device)
│       └── Load model weights onto GPU/CPU
│
├── Set evaluation mode
│   └── model.eval()
│       ├── Disable dropout layers
│       └── Set batch normalization to inference mode
│
└── Store for later use
    ├── self.model = t5_model
    ├── self.tokenizer = t5_tokenizer
    └── logger.info("✓ Generator Ready")
```

---

## PHASE 2: MAIN LOOP - USER QUERY PROCESSING

```
while True:
│
├── INPUT: User provides query
│   └── user_input = input("\n🔍 Enter your question (or 'quit'): ")
│       └── Example: "What is machine learning?"
│
├── CHECK: Is user quitting?
│   ├── if user_input.lower() == "quit":
│   │   └── break (exit loop)
│   │
│   └── else: continue to retrieval
│
├─────────────────────────────────────────────────────────────────────
│ RETRIEVAL PHASE: Find relevant documents
├─────────────────────────────────────────────────────────────────────
│
├── context = retriever.retrieve(user_input, k=3)
│   │   **File:** src/rag/retriever.py → Retriever.retrieve()
│   │
│   ├── Step 1: Embed the query
│   │   └── query_embedding = self.model.encode(user_input)
│   │       ├── Input: "What is machine learning?"
│   │       ├── Output shape: (384,)
│   │       └── Output: [0.123, -0.456, 0.789, ..., 0.234]
│   │
│   ├── Step 2: Convert to numpy array
│   │   └── query_embedding = np.array(embedding).astype('float32')
│   │       └── Required for FAISS operations
│   │
│   ├── Step 3: Search FAISS index
│   │   └── distances, indices = self.index.search(query_embedding, k=3)
│   │       ├── distances = [0.5, 1.2, 2.1]   (L2 distance scores)
│   │       └── indices = [5, 12, 3]          (doc indices)
│   │
│   ├── Step 4: Retrieve document content
│   │   ├── context = ""
│   │   ├── for idx in indices:
│   │   │   ├── doc = self.documents[idx]
│   │   │   └── context += "\n\n" + doc
│   │   │
│   │   └── Result: Top-3 documents concatenated
│   │
│   └── logger.info(f"Retrieved {len(indices)} documents")
│
│   **Context Output Example:**
│   ```
│   Machine learning is a subset of artificial intelligence...
│   
│   Neural networks consist of interconnected layers...
│   
│   Deep learning uses multiple hidden layers...
│   ```
│
├─────────────────────────────────────────────────────────────────────
│ GENERATION PHASE: Generate answer using T5
├─────────────────────────────────────────────────────────────────────
│
├── answer = generator.generate(user_input, context)
│   │   **File:** src/rag/generator.py → Generator.generate()
│   │
│   ├── Step 1: Create prompt
│   │   └── prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
│   │
│   │   **Prompt Output Example:**
│   │   ```
│   │   Question: What is machine learning?
│   │   
│   │   Context:
│   │   Machine learning is a subset of AI that enables systems...
│   │   Neural networks consist of interconnected layers...
│   │   Deep learning uses multiple hidden layers...
│   │   
│   │   Answer:
│   │   ```
│   │
│   ├── Step 2: Tokenize prompt
│   │   ├── inputs = self.tokenizer(prompt, return_tensors="pt")
│   │   │
│   │   ├── Result: Dictionary with:
│   │   │   ├── input_ids: [[101, 3145, 1029, ...]]  (token indices)
│   │   │   └── attention_mask: [[1, 1, 1, ..., 1]]
│   │   │
│   │   └── Move to device
│   │       └── inputs = inputs.to(self.device)  (CPU or GPU)
│   │
│   ├── Step 3: Generate output tokens
│   │   ├── with torch.no_grad():
│   │   │   └── Disable gradient computation (inference only)
│   │   │
│   │   ├── outputs = self.model.generate(
│   │   │       input_ids=input_ids,
│   │   │       attention_mask=attention_mask,
│   │   │       max_length=150,
│   │   │       num_beams=4,
│   │   │       early_stopping=True
│   │   │   )
│   │   │
│   │   └── **T5 Decoding Process:**
│   │       ├── Beam Search (4 beams active)
│   │       │   └── Keep 4 best hypothesis paths simultaneously
│   │       │
│   │       ├── Each generation step:
│   │       │   ├── Model predicts probability of next token
│   │       │   ├── Select top candidates (beam width = 4)
│   │       │   └── Expand each hypothesis
│   │       │
│   │       ├── Stopping criteria:
│   │       │   ├── Max length: 150 tokens
│   │       │   └── Early stop if [EOS] token generated
│   │       │
│   │       └── Result: output_ids = [[0, 105, 3142, 1029, ...]]
│   │
│   ├── Step 4: Decode tokens to text
│   │   ├── answer = self.tokenizer.decode(
│   │   │       output_ids[0],
│   │   │       skip_special_tokens=True
│   │   │   )
│   │   │
│   │   └── Result: "Machine learning is a technique where systems learn..."
│   │
│   └── return answer
│
├─────────────────────────────────────────────────────────────────────
│ OUTPUT PHASE: Display answer
├─────────────────────────────────────────────────────────────────────
│
├── print(f"\n💡 Answer:\n{answer}\n")
│   └── Display generated answer to user terminal
│
└── Loop back to input()
```

---

## PHASE 3: SHUTDOWN

```
User enters 'quit'
│
├── break (exit while True loop)
│
├── logger.info("✓ RAG System shutting down")
│
└── Program ends
```

---

## KEY FILE STRUCTURE & METHODS

### src/main.py

| Component | Purpose |
|-----------|---------|
| `main()` | Entry point that initializes all components |
| `Config()` | Initialize all paths and configuration |
| `Retriever(config)` | Initialize document retriever |
| `Generator(config)` | Initialize answer generator |
| `while True` loop | Main interaction loop for user queries |
| `retriever.retrieve(query)` | Get relevant documents |
| `generator.generate(query, context)` | Generate answer from context |

---

### src/rag/config.py

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__()` | None | Config object | Initialize all paths & settings |
| `_resolve_model_path()` | env_var, local_path, default | str | Find model (env > local > HuggingFace) |
| `_setup_ssl_certificates()` | None | None | Setup macOS SSL certificates |
| `@property emb_model_is_local` | None | bool | Check if embedding model is local |
| `@property gen_model_is_local` | None | bool | Check if generation model is local |

---

### src/rag/retriever.py

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__(config)` | Config object | Retriever object | Load docs & build FAISS index |
| `_load_documents()` | None | None | Load docs from file, split by `---` |
| `_load_embedding_model()` | None | SentenceTransformer | Load `all-MiniLM-L6-v2` |
| `_initialize_faiss_index()` | None | None | Create/load FAISS index |
| `retrieve(query, k=3)` | str, int | str | Embed query, search FAISS, return context |

---

### src/rag/generator.py

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__(config)` | Config object | Generator object | Load T5 tokenizer & model |
| `_load_tokenizer()` | None | T5Tokenizer | Load `t5-small` tokenizer |
| `_load_model()` | None | T5ForConditionalGeneration | Load `t5-small` model |
| `generate(query, context)` | str, str | str | Create prompt → tokenize → generate → decode |

---

## DATA FLOW DIAGRAM

```
User Input
│
├─→ Retriever.retrieve(query)
│   ├─→ Embed query using SentenceTransformer
│   ├─→ Search FAISS index for top-k similar docs
│   └─→ Return concatenated context
│
├─→ Generator.generate(query, context)
│   ├─→ Create prompt: "Question: {query}\nContext: {context}\nAnswer:"
│   ├─→ Tokenize prompt to token IDs
│   ├─→ Run T5 model with beam search (4 beams)
│   └─→ Decode tokens back to text
│
└─→ Print Answer to User Terminal
```

---

## FILES & DEPENDENCIES

### Input Files

| File | Format | Purpose |
|------|--------|---------|
| `data/docs.txt` | Text (split by `---\n`) | Source documents for RAG system |

### Output Files

| File | Format | Purpose |
|------|--------|---------|
| `data/faiss.index` | Binary FAISS format | Pre-computed embedding index |

### Models

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `all-MiniLM-L6-v2` | SentenceTransformer | ~50MB | Embed documents & user queries |
| `t5-small` | Seq2Seq (Tokenizer + Model) | ~250MB | Generate answers from context |

### External Libraries

```
sentence-transformers    → Load sentence embeddings & create vectors
transformers            → Load T5 tokenizer & seq2seq model
faiss-cpu               → Vector search & similarity matching
torch                   → Tensor operations & GPU support
numpy                   → Array operations for embeddings
logging                 → Debug & info output logs
```

---

## EXECUTION FLOW SUMMARY

```
START: main.py
│
├─→ Config.__init__()
│   └─→ Loads paths, resolves models, setup SSL
│
├─→ Retriever(config)
│   └─→ Loads docs, creates embeddings, builds FAISS
│
├─→ Generator(config)
│   └─→ Loads T5 tokenizer & model
│
├─→ ✓ READY FOR QUERIES
│
├─→ [MAIN LOOP] while True:
│   │
│   ├─→ input() ← Get user query
│   │
│   ├─→ retriever.retrieve(query)
│   │   ├─→ Embed query (384D vector)
│   │   ├─→ Search FAISS (L2 distance)
│   │   └─→ Retrieve top-3 documents
│   │
│   ├─→ generator.generate(query, context)
│   │   ├─→ Create prompt
│   │   ├─→ Tokenize
│   │   ├─→ T5 generate (beam search, 4 beams)
│   │   └─→ Decode answer
│   │
│   ├─→ print(answer)
│   │
│   └─→ Loop if not 'quit', else break
│
└─→ [SHUTDOWN]
```

---

## EXAMPLE RUN TRACE

```
User: "What is machine learning?"
│
├─→ retriever.retrieve()
│   ├─→ Embed query
│   │   └─→ [0.123, -0.456, ..., 0.234]  (384D vector)
│   │
│   ├─→ FAISS search
│   │   ├─→ distances: [0.5, 1.2, 2.1]
│   │   └─→ indices: [5, 12, 3]
│   │
│   └─→ Context:
│       "Machine learning is a subset of AI...
│        Neural networks consist of...
│        Deep learning uses multiple..."
│
├─→ generator.generate()
│   ├─→ Create prompt:
│   │   "Question: What is machine learning?
│   │    Context: Machine learning is a subset...
│   │    Answer:"
│   │
│   ├─→ Tokenize: [101, 3145, 1029, ..., 102]
│   │
│   ├─→ T5 Generate (beam search 4 beams)
│   │   └─→ [0, 105, 3142, ...]
│   │
│   └─→ Decode: "Machine learning is a technique where..."
│
└─→ Output:
    "Machine learning is a technique where systems learn from data..."
```

---

**✅ This markdown file is now properly formatted!**

