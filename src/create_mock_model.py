import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import pickle
import os


def create_mock_model(output_dir):
    """
    Creates a mock sklearn pipeline and saves it to mimic PyCaret's output.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Simple pipeline
    model = Pipeline([("classifier", DummyClassifier(strategy="prior"))])

    # Mock fitting
    X = pd.DataFrame(np.random.rand(10, 5), columns=[f"col_{i}" for i in range(5)])
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)

    # Save
    output_path = os.path.join(output_dir, "best_pipeline.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Mock model saved to {output_path}")


if __name__ == "__main__":
    create_mock_model("models")
