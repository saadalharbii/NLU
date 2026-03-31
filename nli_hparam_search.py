{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Demo Notebook for NLI Inference\n",
        "\n",
        "This notebook is the **demo code** for the non-transformer NLI solution in `nli.py`.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Install required packages if needed.\n",
        "# Uncomment the next line and run it on a fresh machine.\n",
        "\n",
        "# %pip install pandas torch scikit-learn\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Runtime inputs\n",
        "\n",
        "Set the file paths below before running the notebook.\n",
        "\n",
        "Only the path to `nli.py` is inferred automatically from the current working directory.\n",
        "The checkpoint path, input CSV path, output CSV path, and optional dev CSV path are passed in here.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "project_dir      = c:\\Users\\fgons\\Downloads\\NLI\n",
            "nli_script_path  = c:\\Users\\fgons\\Downloads\\NLI\\nli.py\n",
            "checkpoint_path  = C:\\Users\\fgons\\Downloads\\NLI\\runs\\checked\\best_model.pt\n",
            "input_csv        = C:\\Users\\fgons\\Downloads\\NLI\\test_data\\test_data\\NLI\\test.csv\n",
            "output_csv       = C:\\Users\\fgons\\Downloads\\NLI\\Group_56_B.csv\n",
            "dev_csv          = None\n",
            "batch_size       = 64\n"
          ]
        }
      ],
      "source": [
        "from pathlib import Path\n",
        "\n",
        "# Assumes this notebook is opened from the same folder as nli.py\n",
        "project_dir = Path.cwd()\n",
        "nli_script_path = project_dir / \"nli.py\"\n",
        "\n",
        "# ---- Pass these values in before running inference ----\n",
        "checkpoint_path = Path(r\"C:\\Users\\fgons\\Downloads\\NLI\\runs\\checked\\best_model.pt\")\n",
        "\n",
        "# Demo input/output paths required by the user\n",
        "input_csv = Path(r\"C:\\Users\\fgons\\Downloads\\NLI\\test_data\\test_data\\NLI\\test.csv\")\n",
        "output_csv = Path(r\"C:\\Users\\fgons\\Downloads\\NLI\\Group_56_B.csv\")\n",
        "\n",
        "# Optional: set this only if you also want to run evaluation on a labelled dev file.\n",
        "# Leave as None to skip evaluation.\n",
        "dev_csv = None\n",
        "\n",
        "batch_size = 64\n",
        "\n",
        "print(\"project_dir      =\", project_dir)\n",
        "print(\"nli_script_path  =\", nli_script_path)\n",
        "print(\"checkpoint_path  =\", checkpoint_path)\n",
        "print(\"input_csv        =\", input_csv)\n",
        "print(\"output_csv       =\", output_csv)\n",
        "print(\"dev_csv          =\", dev_csv)\n",
        "print(\"batch_size       =\", batch_size)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Validate the required paths\n",
        "\n",
        "This cell checks that:\n",
        "- `nli.py` exists\n",
        "- the checkpoint exists\n",
        "- the input CSV exists\n",
        "\n",
        "It also verifies that the required path variables were actually set.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Required files found.\n"
          ]
        }
      ],
      "source": [
        "def _is_placeholder_path(p) -> bool:\n",
        "    if p is None:\n",
        "        return False\n",
        "    text = str(p)\n",
        "    return text.startswith(\"SET_THIS_TO_\")\n",
        "\n",
        "required_values = {\n",
        "    \"nli_script_path\": nli_script_path,\n",
        "    \"checkpoint_path\": checkpoint_path,\n",
        "    \"input_csv\": input_csv,\n",
        "    \"output_csv\": output_csv,\n",
        "}\n",
        "\n",
        "placeholders = [name for name, value in required_values.items() if _is_placeholder_path(value)]\n",
        "if placeholders:\n",
        "    raise ValueError(\n",
        "        \"Please set these path variables in section 1 before running the notebook: \"\n",
        "        + \", \".join(placeholders)\n",
        "    )\n",
        "\n",
        "required_existing = {\n",
        "    \"nli_script_path\": nli_script_path,\n",
        "    \"checkpoint_path\": checkpoint_path,\n",
        "    \"input_csv\": input_csv,\n",
        "}\n",
        "\n",
        "missing = [f\"{name}: {value}\" for name, value in required_existing.items() if not Path(value).exists()]\n",
        "if missing:\n",
        "    raise FileNotFoundError(\"The following required file(s) were not found:\\n- \" + \"\\n- \".join(missing))\n",
        "\n",
        "print(\"Required files found.\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Load the local `nli.py` module\n",
        "\n",
        "This dynamically imports the exact local implementation file so the notebook uses the same saved-model loading and inference functions as the training script.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Loaded: c:\\Users\\fgons\\Downloads\\NLI\\nli.py\n"
          ]
        }
      ],
      "source": [
        "import importlib.util\n",
        "import sys\n",
        "\n",
        "def load_module_from_path(module_name: str, module_path: Path):\n",
        "    module_path = Path(module_path).resolve()\n",
        "    spec = importlib.util.spec_from_file_location(module_name, str(module_path))\n",
        "    if spec is None or spec.loader is None:\n",
        "        raise ImportError(f\"Could not load module from {module_path}\")\n",
        "    module = importlib.util.module_from_spec(spec)\n",
        "    sys.modules[module_name] = module\n",
        "    spec.loader.exec_module(module)\n",
        "    return module\n",
        "\n",
        "nli_module = load_module_from_path(\"user_nli_module\", nli_script_path)\n",
        "print(\"Loaded:\", nli_script_path)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Define notebook wrappers\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {},
      "outputs": [],
      "source": [
        "def run_demo_inference(checkpoint_path, input_csv, output_csv, batch_size=64):\n",
        "    checkpoint_path = str(Path(checkpoint_path))\n",
        "    input_csv = str(Path(input_csv))\n",
        "    output_csv = str(Path(output_csv))\n",
        "\n",
        "    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "    nli_module.run_inference(\n",
        "        checkpoint_path=checkpoint_path,\n",
        "        input_csv=input_csv,\n",
        "        output_csv=output_csv,\n",
        "        batch_size=batch_size,\n",
        "    )\n",
        "\n",
        "    return Path(output_csv)\n",
        "\n",
        "\n",
        "def run_demo_evaluation(checkpoint_path, dev_csv, batch_size=64):\n",
        "    if dev_csv is None:\n",
        "        print(\"dev_csv is None, so evaluation was skipped.\")\n",
        "        return\n",
        "\n",
        "    checkpoint_path = str(Path(checkpoint_path))\n",
        "    dev_csv = str(Path(dev_csv))\n",
        "\n",
        "    nli_module.evaluate_checkpoint(\n",
        "        checkpoint_path=checkpoint_path,\n",
        "        dev_csv=dev_csv,\n",
        "        batch_size=batch_size,\n",
        "    )\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Optional evaluation cell\n",
        "\n",
        "Run this only if you want to evaluate the saved checkpoint on a labelled development file.\n",
        "If `dev_csv` is `None`, the cell will skip evaluation.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "dev_csv is None, so evaluation was skipped.\n"
          ]
        }
      ],
      "source": [
        "run_demo_evaluation(\n",
        "    checkpoint_path=checkpoint_path,\n",
        "    dev_csv=dev_csv,\n",
        "    batch_size=batch_size,\n",
        ")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 6. Run inference on the test input file\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Saved predictions to C:\\Users\\fgons\\Downloads\\NLI\\Group_56_B.csv\n",
            "Done. Predictions saved to: C:\\Users\\fgons\\Downloads\\NLI\\Group_56_B.csv\n"
          ]
        }
      ],
      "source": [
        "saved_output = run_demo_inference(\n",
        "    checkpoint_path=checkpoint_path,\n",
        "    input_csv=input_csv,\n",
        "    output_csv=output_csv,\n",
        "    batch_size=batch_size,\n",
        ")\n",
        "\n",
        "print(f\"Done. Predictions saved to: {saved_output}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 7. Preview the predictions file\n",
        "\n",
        "The current `nli.py` implementation writes a CSV with a single `label` column.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Predictions file shape: (3302, 1)\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   label\n",
              "0      1\n",
              "1      0\n",
              "2      1\n",
              "3      0\n",
              "4      1"
            ]
          },
          "execution_count": 14,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "preds_df = pd.read_csv(saved_output)\n",
        "print(\"Predictions file shape:\", preds_df.shape)\n",
        "preds_df.head()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Example submission filename for this notebook's solution:\n",
        "# output_csv = Path(r\"Group_12_B.csv\")\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": ".venv (3.14.0)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.14.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
