import os
from dotenv import load_dotenv
import dsp
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# Load environment variables
load_dotenv()
unify_api_key = os.getenv("UNIFY_KEY")

# Initialize Unify model
model = dsp.Unify(
    model="gpt-3.5-turbo@openai",
    max_tokens=150,
    stream=True,
    api_key=unify_api_key,
)
dspy.settings.configure(lm=model)

# Load GSM8K dataset
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

print("Loading GSM8K train and dev sets")
print(f"Train set: {gsm8k_trainset}")

# Define Chain of Thought (CoT) module
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question) -> str:
        return self.prog(question=question)

# Configure BootstrapFewShot optimizer
config = {"max_bootstrapped_demos": 4, "max_labeled_demos": 4}
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)

# Optimize CoT program using train set
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

# Set up evaluator
evaluate = Evaluate(
    devset=gsm8k_devset,
    metric=gsm8k_metric,
    num_threads=4,
    display_progress=True,
    display_table=0
)

# Evaluate optimized CoT program
evaluate(optimized_cot)
model.inspect_history(n=1)

print(
    """Done! This example showcases how to set up your environment, define a custom module,
    compile a model, and rigorously evaluate its performance using the provided dataset and teleprompter configurations."""
)
