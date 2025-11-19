
# 🧠 Multi-Agent Product Matcher (Ollama)

Questo progetto implementa un sistema **multi-agente** per analizzare e validare automaticamente 
le corrispondenze tra prodotti OCR (letti da scontrini) e un catalogo promozionale, 
utilizzando modelli **LLM** (tramite Ollama).

---

## 📁 Struttura del Progetto

```
multi_agent_matcher/
│
├── classes/
│   ├── __init__.py
│   └── LLM.py                   # Wrapper per Ollama LLM
│
├── agents/
│   ├── __init__.py
│   ├── base.py                  # Classe base Agent
│   └── prompt_templates.py      # Prompt per Analista, Critico e Arbitro
│
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Funzioni di supporto per NDJSON e snapshot
│
├── data/
│   ├── prodotti.csv             # Lista prodotti in promozione
│   └── data.csv                 # Dati OCR da scontrini
│
├── results/
│   ├── matches.ndjson           # Output incrementale (append)
│   ├── matches.json             # Output finale (snapshot completo)
│   └── predizioni_simplified.json  # Output leggibile (descrizione + match finale)
│
├── multi_agent_matcher.py       # Script principale di orchestrazione
├── README.md                    # Documentazione del progetto
└── requirements.txt             # Librerie richieste
```

---

## 🤖 Architettura Multi-Agente

Il sistema è composto da **tre agenti principali** che cooperano in sequenza:

| Agente | Ruolo | Output |
|--------|--------|--------|
| 🧩 **Analista** | Propone i 3 migliori match basati sulla descrizione OCR | JSON con id, brand, nome, confidenza, spiegazione |
| 🧠 **Critico** | Valuta la coerenza e la sicurezza della risposta dell’Analista | JSON con valutazione e affidabilità |
| ⚖️ **Arbitro** | Decide se accettare, rifiutare o revisionare il risultato | JSON con decisione finale, motivazione e match finale |

---

## ⚙️ Requisiti

- Python 3.10+
- [Ollama](https://ollama.ai) installato e configurato localmente
- Modello consigliato: `mistral:7b`

### 📦 Installazione
```bash
pip install -r requirements.txt
```

---

## 🚀 Esecuzione

Per avviare il sistema:
```bash
python multi_agent_matcher.py
```

I risultati saranno salvati nella cartella `results/`.

---

## 📊 Output Files

| File | Descrizione |
|------|--------------|
| `results/matches.ndjson` | Log incrementale (una riga JSON per ogni record processato) |
| `results/matches.json` | Snapshot completo in formato JSON strutturato |
| `results/predizioni_simplified.json` | Risultati minimali (descrizione OCR + match finale) |

---

## 🧩 Espansioni future

- Integrazione con **LangGraph** per un workflow multi-agente visivo
- Dashboard in **Streamlit** per visualizzare le decisioni
- Supporto per **modelli multimodali (immagini + testo)**

---

## ✍️ Autore
Progetto sviluppato come esempio di architettura multi-agente con Ollama e LLM locali.
