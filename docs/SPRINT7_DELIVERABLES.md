# Sprint 7 Deliverables

This document contains the deliverables for the final sprint of the Heart Disease Prediction project.

## 1. Code Quality Checklist (Issue 45)

### Verification Report
- **Structure**: The `src/` directory has been refactored.
- **Docstrings**: Added Google-style docstrings to `src/app.py`, `src/train_pycaret.py`, and `src/data_ingestion.py`.
- **Type Hinting**: Added static typing (e.g., `-> Optional[pd.DataFrame]`) to critical functions.
- **Cleanliness**:
  - Removed unused imports (e.g., `pickle`, `traceback`).
  - Removed dead code (commented out paths and debug logic).
  - Standardized import ordering.

### Recommendations for Future Sprints
- Implement `mypy` in the CI/CD pipeline to enforce type safety.
- Convert `print` statements in `train_pycaret.py` and `data_ingestion.py` to `logging` module calls for better traceability.
- Add unit tests for `data_ingestion.py` using a mock XPT file generator.

---

## 2. Presentation Outline (Issue 46)

### Title: "HeartGuard AI: From Mathematical First Principles to Ethical Production"
**Duration:** 15 Minutes

#### Slide 1: The Hook (1 min)
*   **Visual:** A beating heart monitor flatlining.
*   **Narrative:** "Cardiovascular diseases claim 17.9 million lives a year. That's one death every 2 seconds. Today, we present not just a prediction model, but a life-saving assistant."
*   **Key Data:** Leading cause of death globally (WHO).

#### Slide 2: The Journey - "Why Re-invent the Wheel?" (3 mins)
*   **Visual:** Split screen. Left: "Academic (Scratch)", Right: "Production (PyCaret)".
*   **Narrative:** "We didn't just `import xgboost`. We built it from scratch to understand the math ($Gain = \frac{1}{2} [\frac{G_L^2}{H_L+\lambda} + ...]$). Then, we leveraged PyCaret for robust, scalable deployment."
*   **Key Point:** Deep understanding leads to better tuning and debugging.

#### Slide 3: The Engine - Metrics that Matter (2 mins)
*   **Visual:** Confusion Matrix highlighting False Negatives vs. False Positives.
*   **Narrative:** "In healthcare, missing a sick patient (False Negative) is fatal. We optimized for **Recall** and **F2-Score**, sacrificing some precision to save more lives."
*   **Key Metric:** Recall > 0.85 (Target).

#### Slide 4: Value Added - "Not a Black Box" (3 mins)
*   **Visual:** Screenshot of SHAP Waterfall plot from the UI.
*   **Narrative:** "Doctors don't trust black boxes. Our system explains *why*: 'Your risk is high *because* your BMI is 35 and you smoke', not just 'Probability: 80%'."
*   **Demo Tease:** "We'll see this live in a moment."

#### Slide 5: Responsibility - Ethical AI (2 mins)
*   **Visual:** Fairlearn charts showing parity between Gender/Age groups.
*   **Narrative:** "A model that only works for one demographic is dangerous. We audited our model for bias ensuring fair predictions across gender and age groups."

#### Slide 6: Live Demo (3 mins)
*   (Switch to Streamlit tab)
*   **Action:** Walkthrough of Happy Path vs. Risk Path.

#### Slide 7: Future & Conclusion (1 min)
*   **Visual:** Roadmap graphic.
*   **Narrative:** "Next steps: Integration with hospital EMR systems, real-time wearable data, and federated learning for privacy."
*   **Closing:** "HeartGuard AI: Technology with a heartbeat."

---

## 3. Demo Script (Issue 47)

**Presenter:** [Your Name]
**Setup:** Streamlit app running locally, browser open at `http://localhost:8501`.

### Phase 1: Introduction (0:00 - 0:30)
*   **Say:** "Welcome to the HeartGuard interface. Designed for clinicians, it's clean, fast, and informative."
*   **Action:** Point out the Sidebar warning ("Support tool, not replacement").

### Phase 2: Happy Path (0:30 - 1:15)
*   **Say:** "Let's simulate a healthy patient: 'Jane', 30 years old, active."
*   **Action:**
    *   **Age:** Set to **30**.
    *   **BMI:** Set to **22.0**.
    *   **Sex:** Select **Female**.
    *   **Smoker:** **No**.
    *   **Diabetes:** **No**.
    *   **Physical Activity:** **Yes**.
    *   **Threshold:** Leave at default (optimized).
*   **Click:** "Predict Risk".
*   **Say:** "Instantly, we see a **Low Risk** signal. The probability is minimal."

### Phase 3: Risk Path & SHAP (1:15 - 2:30)
*   **Say:** "Now, let's look at a high-risk case: 'John', 65, heavy smoker."
*   **Action:**
    *   **Age:** Change to **65**.
    *   **BMI:** Change to **31.0** (Obese).
    *   **Sex:** Change to **Male**.
    *   **Smoker:** Change to **Yes**.
    *   **Diabetes:** Change to **Yes**.
*   **Click:** "Predict Risk".
*   **Say:** "The model flags **High Risk**."
*   **Action:** Check the box **"Ver por qué el modelo tomó esta decisión"**.
*   **Say:** "Now, the magic. While the explanation loads..." *(Transition Phrase)* "...calculating Shapley values requires analyzing thousands of decision trees to attribute contribution scores to each feature."
*   **Action:** Point to Waterfall plot.
*   **Say:** "Here we see that Age and Smoking Status were the biggest drivers pushing the risk up (red bars), while Physical Activity tried to lower it (blue bars)."

### Phase 4: Closing (2:30 - 3:00)
*   **Say:** "This transparency empowers doctors to have evidence-based conversations with patients about lifestyle changes."

**Backup Plan:**
*   *If Streamlit crashes:* "Live demos are always fun! Let me switch to a pre-recorded walkthrough of this exact scenario." (Have video `demo_backup.mp4` ready in QuickTime/VLC).

---

## 4. Final Delivery Checklist (Issue 48)

### Repository Structure
- [ ] **README.md**: Create with Badge, GIF, Install, Structure.
- [ ] **LICENSE**: Add MIT License.
- [ ] **.gitignore**: Ensure `data/`, `models/*.pkl`, `__pycache__` are ignored.
- [ ] **requirements.txt**: Freeze current environment.

### Code Quality
- [x] **Refactoring**: `src/` files cleaned and documented.
- [ ] **Tests**: Run `pytest` to ensure no regressions.
- [ ] **Formatting**: Run `black .` and `ruff .` (if installed) or manual check.

### Artifacts
- [ ] **Model**: Ensure `models/final_pipeline_v1.pkl` and `model_config.json` are present (or instructions to generate them).
- [ ] **Data**: `data/01_raw/` should contain the sample or instructions to download.

### Documentation
- [x] **Deliverables**: This document (`SPRINT7_DELIVERABLES.md`) created.
- [ ] **Wiki/Docs**: Ensure `docs/` folder is up to date.

### Version Control
- [ ] **Tag**: Create git tag `v1.0` after final commit.
