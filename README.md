# CARDIA

### Cardiovascular Digital Twin • Experiment Engine • 3D Heart • ML • RAG

**CARDIA turns cardiovascular physiology into an experiment.**

CARDIA is an interactive cardiovascular digital twin that allows users to create a virtual patient, perturb physiological parameters, observe the cardiovascular system respond in real time, run alternative interventions, compare counterfactual outcomes, and understand the underlying physiological mechanisms through evidence-grounded explanations.

Instead of simply displaying what a heart is doing, CARDIA lets you **experiment on a virtual patient**.

---

## Why CARDIA?

Most cardiovascular simulators focus on visualization or parameter manipulation.

CARDIA is designed around a different interaction:

> **Observe → Perturb → Simulate → Compare → Explain**

A user can:

* Modify heart rate, blood volume, systemic vascular resistance, or contractility.
* Watch the virtual heart respond in real time.
* Observe changes in pressure, volume, cardiac output, stroke volume, and ECG.
* Run physiological scenarios such as hemorrhage, hypertension, and tachycardia.
* Fork the same patient into different intervention scenarios.
* Compare what happens with and without an intervention.
* Use machine learning to estimate hidden physiological parameters of a synthetic patient.
* Ask why a physiological change occurred and receive a mechanism-based explanation grounded in curated sources.

---

# Core Concept

```text
                    VIRTUAL PATIENT
                           │
                           ▼
                  ┌─────────────────┐
                  │ PHYSIOLOGICAL   │
                  │    STATE        │
                  └────────┬────────┘
                           │
                    USER PERTURBATION
                           │
                           ▼
                  ┌─────────────────┐
                  │ CARDIOVASCULAR  │
                  │ SIMULATION      │
                  │     ENGINE      │
                  └────────┬────────┘
                           │
                ┌──────────┼──────────┐
                ▼          ▼          ▼
             3D Heart    Metrics     ECG
                │          │          │
                └──────────┼──────────┘
                           │
                           ▼
                    EXPERIMENT ENGINE
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              INTERVENTION    NO INTERVENTION
                    │             │
                    └──────┬──────┘
                           ▼
                    COUNTERFACTUAL
                      COMPARISON
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
             ML                        RAG
       Hidden parameter          Mechanistic
          estimation             explanation
```

---

# Key Features

## 1. Interactive Cardiovascular Digital Twin

CARDIA models a simplified cardiovascular system containing:

* Right atrium
* Right ventricle
* Left atrium
* Left ventricle
* Pulmonary circulation
* Systemic circulation
* Cardiac valves
* Arterial and venous compartments

The simulation continuously updates the cardiovascular state as the virtual patient changes.

---

## 2. Real-Time 3D Heart

The cardiovascular state drives an interactive 3D heart.

Simulation variables are mapped directly to visual behavior:

| Physiological variable | Visualization            |
| ---------------------- | ------------------------ |
| Heart rate             | Beat frequency           |
| Chamber volume         | Chamber filling/emptying |
| Contractility          | Contraction intensity    |
| Valve state            | Valve animation          |
| Blood flow             | Flow particles           |
| Pressure               | Real-time graphs         |
| Electrical activity    | ECG visualization        |

The goal is not simply to display a heart model, but to make the consequences of physiological changes visually observable.

---

# 3. Cardiovascular Simulation Engine

CARDIA uses a deterministic physiological model as its source of truth.

Core relationships include:

```text
Stroke Volume = EDV − ESV

Cardiac Output = Heart Rate × Stroke Volume

MAP ≈ Cardiac Output × SVR

Flow = ΔP / R
```

The model incorporates simplified representations of:

* Preload
* Afterload
* Contractility
* Heart rate
* Blood volume
* Systemic vascular resistance
* Venous return
* Frank-Starling response
* Baroreflex feedback
* Valve dynamics

The simulation engine produces a shared `SimulationState` consumed by the rest of the system.

---

# 4. Experiment Engine

CARDIA is built around experimentation rather than passive visualization.

Users can perturb the virtual patient and observe the resulting physiological cascade.

### Example: Hemorrhage

```text
Blood Volume ↓
      ↓
Venous Return ↓
      ↓
EDV ↓
      ↓
Stroke Volume ↓
      ↓
Cardiac Output ↓
      ↓
Blood Pressure ↓
      ↓
Baroreflex Response
      ↓
Heart Rate ↑
```

The entire cascade can be observed through the 3D heart, ECG, metrics, and graphs.

---

# 5. Counterfactual Simulation

One of CARDIA's core capabilities is the ability to compare alternative futures from the same physiological state.

```text
                    SAME PATIENT
                         │
                  ┌──────┴──────┐
                  ▼             ▼
            NO INTERVENTION   INTERVENTION
                  │             │
                  ▼             ▼
             Simulation A   Simulation B
                  │             │
                  └──────┬──────┘
                         ▼
                    COMPARISON
```

For example:

```text
                    No Action       Intervention

MAP                   58                 76
Cardiac Output        3.1                4.6
Stroke Volume         41                 57
Heart Rate           108                 91
```

CARDIA can then visualize the divergence between the two simulated trajectories.

This turns the system from a simple simulator into an **experimental and counterfactual environment**.

---

# 6. Machine Learning

CARDIA uses machine learning to estimate latent physiological parameters from synthetic observations.

### Pipeline

```text
Synthetic Patient
       ↓
Physiological Simulation
       ↓
Synthetic Observations
       ↓
Feature Extraction
       ↓
ML Model
       ↓
Estimated Hidden Parameters
```

Potential inputs include:

* Heart rate
* Blood pressure
* ECG-derived features
* Waveform features
* Other simulated physiological observations

Potential outputs include:

* Estimated blood volume
* Contractility
* Systemic vascular resistance
* Other latent simulation parameters

The ML model is trained and evaluated on synthetic patients generated by the mechanistic simulation.

**CARDIA does not claim clinical diagnostic accuracy.**

---

# 7. Evidence-Grounded AI Explanations

CARDIA combines the current simulation state with a retrieval-based physiology knowledge layer.

```text
User Question
      +
Current Simulation State
      ↓
Knowledge Retrieval
      ↓
Relevant Physiology Sources
      ↓
LLM
      ↓
Mechanistic Explanation
      +
Source References
```

Example:

> **Why did blood pressure decrease?**

CARDIA can trace the simulated mechanism:

```text
Blood Volume ↓
→ Venous Return ↓
→ EDV ↓
→ Stroke Volume ↓
→ Cardiac Output ↓
→ MAP ↓
```

The language model explains the mechanism using retrieved physiological evidence.

### Important architectural principle

**The LLM does not calculate the physiology.**

The simulation engine calculates the state.

The retrieval system provides supporting knowledge.

The LLM explains the result.

---

# Architecture

```text
                         ┌──────────────────────┐
                         │       FRONTEND       │
                         │ Next.js + TypeScript  │
                         │ Three.js / R3F        │
                         └──────────┬───────────┘
                                    │
                              WebSocket / API
                                    │
                         ┌──────────▼───────────┐
                         │        FASTAPI       │
                         │   API + Orchestration│
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │ PHYSIOLOGY   │      │     ML       │      │     RAG      │
      │    ENGINE    │      │   ENGINE     │      │  + LLM       │
      └──────┬───────┘      └──────────────┘      └──────────────┘
             │
             ▼
      ┌─────────────────┐
      │ EXPERIMENT      │
      │ ENGINE           │
      │                 │
      │ Counterfactuals │
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │    SUPABASE     │
      │ Patients        │
      │ Sessions        │
      │ Experiments     │
      │ Results         │
      │ Predictions     │
      │ Explanations    │
      └─────────────────┘
```

---

# Tech Stack

## Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* Three.js
* React Three Fiber
* Recharts / Plotly

## Backend

* Python
* FastAPI
* Pydantic
* WebSockets
* AsyncIO

## Simulation

* Python
* NumPy
* SciPy
* Pandas
* Pytest

## Machine Learning

* PyTorch
* Scikit-learn
* NumPy
* Pandas

## RAG / AI

* LlamaIndex
* Qdrant
* Embeddings
* LLM API

## Database

* Supabase
* PostgreSQL
* JSONB

## Infrastructure

* GitHub
* GitHub Actions
* Vercel
* FastAPI deployment
* Optional Docker

---

# Database

CARDIA uses Supabase for persistent application data.

```text
patients
    │
    ▼
simulation_sessions
    │
    ├──────────────► experiments
    │                    │
    │                    ▼
    │             experiment_results
    │
    ├──────────────► counterfactual_runs
    │
    ├──────────────► ml_predictions
    │
    └──────────────► explanations
```

### `patients`

Stores virtual patient parameters.

### `simulation_sessions`

Stores simulation sessions and important state checkpoints.

### `experiments`

Stores user interventions and parameter changes.

### `experiment_results`

Stores baseline/result states and simulation outcomes.

### `counterfactual_runs`

Stores alternative simulation trajectories and comparisons.

### `ml_predictions`

Stores latent parameter estimates and model metadata.

### `explanations`

Stores simulation-aware AI explanations and retrieved evidence.

High-frequency simulation states are streamed in memory through WebSockets rather than written to the database every timestep.

---

# API

Example endpoints:

```text
POST /patients
GET  /patients

POST /simulation/start
POST /simulation/pause
POST /simulation/reset

POST /experiment/apply

POST /counterfactual/fork

POST /ml/infer

POST /explain

WS   /simulation/stream
```

---

# Shared Simulation State

All major components communicate through a common state representation.

```json
{
  "patient_id": "patient-001",
  "time": 12.42,

  "hr": 82,

  "bp": {
    "sys": 112,
    "dia": 70,
    "map": 84
  },

  "cardiac": {
    "edv": 118,
    "esv": 55,
    "stroke_volume": 63,
    "cardiac_output": 5.17,
    "contractility": 1.0
  },

  "circulation": {
    "blood_volume": 5.1,
    "svr": 1.1
  },

  "valves": {
    "mitral": false,
    "aortic": true,
    "tricuspid": false,
    "pulmonary": false
  }
}
```

This contract allows the simulation, frontend, ML, RAG, and backend teams to work independently.

---

# Project Structure

```text
cardia/
│
├── frontend/
│   ├── components/
│   ├── heart/
│   ├── charts/
│   ├── experiments/
│   └── app/
│
├── backend/
│   ├── api/
│   ├── websocket/
│   ├── services/
│   └── schemas/
│
├── simulation/
│   ├── cardiovascular.py
│   ├── chambers.py
│   ├── circulation.py
│   ├── valves.py
│   ├── feedback.py
│   ├── scenarios.py
│   └── tests/
│
├── ml/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── inference/
│
├── rag/
│   ├── ingestion/
│   ├── retrieval/
│   ├── prompts/
│   └── sources/
│
├── database/
│   ├── schema.sql
│   └── seed.sql
│
├── tests/
│
└── docs/
    ├── architecture.md
    ├── physiology.md
    └── api.md
```

---

# Development Workflow

CARDIA is developed as parallel workstreams.

```text
Person 1 → Physiology Engine
Person 2 → 3D + Frontend
Person 3 → Backend + Supabase + Integration
Person 4 → ML
Person 5 → RAG + AI
```

All components communicate through defined interfaces rather than tightly coupled implementations.

### Branches

```text
main

feature/physiology-engine
feature/heart-3d
feature/backend
feature/ml
feature/rag
```

### Integration principle

Build against mocks first.

For example:

```text
Frontend
   ↓
Mock SimulationState
```

can later become:

```text
Frontend
   ↓
WebSocket
   ↓
Real SimulationState
```

without rewriting the frontend.

---

# Demo Flow

CARDIA's intended demonstration follows a single continuous experiment.

### 01 — Meet the patient

Start with a healthy virtual patient.

### 02 — Break the patient

Reduce blood volume.

The audience watches:

* chamber filling decrease
* stroke volume change
* cardiac output fall
* blood pressure fall
* heart rate compensate
* ECG change

### 03 — Ask why

Ask:

> "Why did this happen?"

CARDIA explains the physiological cascade using retrieved evidence.

### 04 — Intervene

Apply an intervention to the virtual patient.

### 05 — Explore the alternative future

Run the same physiological state through two trajectories:

```text
No Intervention
        VS
Intervention
```

### 06 — Compare

Visualize how the cardiovascular state diverges.

### 07 — Investigate

Use ML to estimate hidden physiological parameters from simulated observations.

---

# Design Philosophy

CARDIA follows three principles.

### 1. Simulation before AI

The physiological model is the source of truth.

### 2. Experiment before prediction

CARDIA is designed around manipulating a virtual patient and observing consequences.

### 3. Explanation after evidence

AI explanations are grounded in both the current simulation state and retrieved physiology knowledge.

---

# Medical Scope

CARDIA is an **educational and research-oriented cardiovascular simulation platform**.

It uses synthetic virtual patients and simplified physiological models.

It is **not**:

* a medical device
* a diagnostic system
* a treatment recommendation system
* a substitute for clinical judgment
* validated for real-patient decision making

Any ML results are estimates within the synthetic simulation environment.

---

# Hackathon Objective

CARDIA is being developed for **Hackulus '26** with a focus on:

* Technical depth
* Real-time interaction
* Physiological modeling
* Multidisciplinary engineering
* Explainable AI
* Counterfactual experimentation
* Real-world educational/research value
* High-quality user experience

---

# Status

🚧 **Active Development**

Current priorities:

* [ ] Cardiovascular simulation engine
* [ ] Real-time 3D heart
* [ ] ECG visualization
* [ ] Experiment engine
* [ ] Counterfactual simulations
* [ ] Synthetic patient generation
* [ ] ML parameter inference
* [ ] RAG knowledge system
* [ ] Simulation-aware AI explanations
* [ ] Supabase persistence
* [ ] End-to-end deployment

---

# Team

Built by a multidisciplinary team combining:

* Physiological modeling
* Full-stack engineering
* 3D graphics
* Machine learning
* Retrieval-augmented generation
* Systems integration

---

## CARDIA

> **Observe. Perturb. Simulate. Compare. Explain.**

**CARDIA turns cardiovascular physiology into an experiment.**
