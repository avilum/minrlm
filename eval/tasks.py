"""
Task Definitions for RLM Evaluation

Each task is a standardized benchmark that tests specific capabilities:
- S-NIAH: Single needle-in-a-haystack retrieval
- Multi-Needle: Multiple item retrieval
- OOLONG-Pairs: Complex matching across long context
- Scaling: Context length stress test

To add a new task, subclass BaseTask and use @register_task decorator.
"""

import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Task registry for extensibility
TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Decorator to register a task class."""

    def decorator(cls):
        TASK_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


@dataclass
class TaskInstance:
    """A single task instance with context and expected answer."""

    task: str  # The question/instruction
    context: str  # The context/haystack
    expected: str  # Expected answer for validation
    metadata: dict | None = None  # Task-specific metadata


class BaseTask(ABC):
    """Base class for evaluation tasks."""

    name: str = "base"
    description: str = "Base task"
    difficulty: str = "medium"

    @abstractmethod
    def generate(self, seed: int = 42, **kwargs) -> TaskInstance:
        """Generate a task instance."""
        pass

    @abstractmethod
    def check(self, response: str, expected: str) -> bool:
        """Check if response is correct."""
        pass

    def check_partial(self, response: str, expected: str) -> float:
        """Return partial score (0.0-1.0). Default: binary."""
        return 1.0 if self.check(response, expected) else 0.0


# =============================================================================
# Utility Functions
# =============================================================================


def generate_noise(length: int, seed: int = 42) -> str:
    """Generate plausible-looking filler text."""
    random.seed(seed)
    words = [
        "data",
        "analysis",
        "result",
        "process",
        "study",
        "algorithm",
        "function",
        "method",
        "system",
        "network",
        "model",
        "output",
        "input",
        "value",
        "parameter",
        "variable",
        "structure",
        "pattern",
        "research",
        "finding",
        "evidence",
        "theory",
        "hypothesis",
        "approach",
        "computation",
        "observation",
        "measurement",
        "experiment",
        "conclusion",
    ]

    text = []
    current_len = 0
    while current_len < length:
        # Create varied sentence lengths
        sent_len = random.randint(8, 20)
        sentence = " ".join(random.choices(words, k=sent_len))
        text.append(sentence.capitalize() + ". ")
        current_len += len(text[-1])

    return "".join(text)[:length]


def generate_code(prefix: str = "CODE", length: int = 8, seed: int = None) -> str:
    """Generate a random alphanumeric code."""
    if seed is not None:
        random.seed(seed)
    chars = "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{prefix}-{chars}"


# =============================================================================
# Task Implementations
# =============================================================================


@register_task("sniah")
class SNIAHTask(BaseTask):
    """
    S-NIAH: Single Needle-in-a-Haystack

    Find a single secret code hidden in a large context.
    This tests basic long-context retrieval capabilities.
    """

    description = "Find a single secret code hidden in noise text"
    difficulty = "easy"

    def generate(self, seed: int = 42, context_size: int = 50000, **kwargs) -> TaskInstance:
        random.seed(seed)

        # Generate unique secret
        secret = generate_code("SECRET", 8, seed + 1)
        needle = f"\n\n[IMPORTANT: The secret code is: {secret}]\n\n"

        # Create haystack with needle inserted randomly
        noise = generate_noise(context_size, seed + 2)
        position = random.randint(len(noise) // 4, 3 * len(noise) // 4)
        context = noise[:position] + needle + noise[position:]

        task = """Find the secret code hidden in this text.
The text contains a line like: "[IMPORTANT: The secret code is: SECRET-XXXXXXXX]"
Return ONLY the full secret code (e.g., SECRET-ABC12345)."""

        return TaskInstance(
            task=task,
            context=context,
            expected=secret,
            metadata={"context_size": context_size, "needle_position": position},
        )

    def check(self, response: str, expected: str) -> bool:
        # Extract the code part for flexible matching
        code = expected.split("-", 1)[1] if "-" in expected else expected
        return code.upper() in response.upper()


@register_task("multi_needle")
class MultiNeedleTask(BaseTask):
    """
    Multi-Needle S-NIAH

    Find multiple secret codes scattered throughout context.
    Tests ability to maintain attention across multiple targets.
    """

    description = "Find multiple secret codes scattered in text"
    difficulty = "medium"

    def generate(self, seed: int = 42, context_size: int = 50000, num_needles: int = 5, **kwargs) -> TaskInstance:
        random.seed(seed)

        # Generate unique secrets
        secrets = [generate_code("NEEDLE", 6, seed + i * 10) for i in range(num_needles)]

        # Create haystack
        noise = generate_noise(context_size, seed + 100)
        context = noise

        # Insert needles at random positions
        positions = sorted(random.sample(range(len(noise) // 10, 9 * len(noise) // 10), num_needles))

        offset = 0
        for i, pos in enumerate(positions):
            needle = f"\n\n[SECRET #{i + 1}: {secrets[i]}]\n\n"
            insert_pos = pos + offset
            context = context[:insert_pos] + needle + context[insert_pos:]
            offset += len(needle)

        task = f"""Find ALL {num_needles} secret codes hidden in this text.
Each secret appears as: "[SECRET #N: NEEDLE-XXXXXX]"
Return ALL codes as a comma-separated list (e.g., NEEDLE-ABC123, NEEDLE-DEF456, ...)."""

        expected = ", ".join(secrets)

        return TaskInstance(
            task=task,
            context=context,
            expected=expected,
            metadata={
                "context_size": context_size,
                "num_needles": num_needles,
                "secrets": secrets,
            },
        )

    def check(self, response: str, expected: str) -> bool:
        """Check if at least 80% of needles found."""
        secrets = [s.strip() for s in expected.split(",")]
        found = sum(1 for s in secrets if s.split("-", 1)[1].upper() in response.upper())
        return found >= len(secrets) * 0.8

    def check_partial(self, response: str, expected: str) -> float:
        """Return fraction of needles found."""
        secrets = [s.strip() for s in expected.split(",")]
        found = sum(1 for s in secrets if s.split("-", 1)[1].upper() in response.upper())
        return found / len(secrets) if secrets else 0.0


@register_task("pairs")
class PairsTask(BaseTask):
    """
    OOLONG-Pairs

    Match definitions to concepts scattered across context.
    This is the hardest task - "even frontier models fail catastrophically" (paper).
    """

    description = "Match scattered definitions to concepts"
    difficulty = "hard"

    def generate(self, seed: int = 42, context_size: int = 50000, num_pairs: int = 8, **kwargs) -> TaskInstance:
        random.seed(seed)

        # Greek letters and scientific concepts
        greek = [
            "ALPHA",
            "BETA",
            "GAMMA",
            "DELTA",
            "EPSILON",
            "ZETA",
            "ETA",
            "THETA",
            "IOTA",
            "KAPPA",
            "LAMBDA",
            "MU",
            "NU",
            "XI",
            "OMICRON",
            "PI",
        ]
        concepts = [
            "quantum entanglement",
            "photosynthesis",
            "continental drift",
            "neural plasticity",
            "entropy",
            "fibonacci sequence",
            "golden ratio",
            "prime numbers",
            "antimatter",
            "superconductivity",
            "black holes",
            "dark matter",
            "evolution",
            "relativity",
            "electromagnetism",
            "thermodynamics",
        ]

        random.shuffle(greek)
        random.shuffle(concepts)

        pairs = [(f"DEF-{greek[i]}", concepts[i]) for i in range(num_pairs)]

        # Create context with scattered definitions
        noise = generate_noise(context_size, seed + 1)
        context = noise

        # Insert definitions in first half
        def_positions = sorted(random.sample(range(len(noise) // 10, len(noise) // 2), num_pairs))

        offset = 0
        for i, pos in enumerate(def_positions):
            definition = f"\n\n[DEFINITION {pairs[i][0]}]: The concept of {pairs[i][1]}.\n\n"
            insert_pos = pos + offset
            context = context[:insert_pos] + definition + context[insert_pos:]
            offset += len(definition)

        task = f"""Match each definition code to its concept. The text contains:
- {num_pairs} definitions like "[DEFINITION DEF-XXXX]: The concept of <concept>."

Return ALL {num_pairs} pairs in format: DEF-XXXX=concept, DEF-YYYY=concept, ..."""

        expected = ", ".join([f"{code}={concept}" for code, concept in pairs])

        return TaskInstance(
            task=task,
            context=context,
            expected=expected,
            metadata={
                "context_size": context_size,
                "num_pairs": num_pairs,
                "pairs": pairs,
            },
        )

    def check(self, response: str, expected: str) -> bool:
        """Check if at least 80% of pairs matched correctly."""
        pairs = [p.strip() for p in expected.split(",")]
        found = 0
        for pair in pairs:
            if "=" in pair:
                code, concept = pair.split("=", 1)
                if code.upper() in response.upper() and concept.lower() in response.lower():
                    found += 1
        return found >= len(pairs) * 0.8

    def check_partial(self, response: str, expected: str) -> float:
        """Return fraction of pairs matched."""
        pairs = [p.strip() for p in expected.split(",")]
        found = 0
        for pair in pairs:
            if "=" in pair:
                code, concept = pair.split("=", 1)
                if code.upper() in response.upper() and concept.lower() in response.lower():
                    found += 1
        return found / len(pairs) if pairs else 0.0


@register_task("scaling")
class ScalingTask(BaseTask):
    """
    Context Scaling Test

    Test S-NIAH at various context lengths to measure degradation.
    Replicates Figure 1 from the paper.
    """

    description = "S-NIAH across different context sizes"
    difficulty = "variable"

    def __init__(self, context_size: int = 50000):
        self.context_size = context_size

    def generate(self, seed: int = 42, context_size: int = None, **kwargs) -> TaskInstance:
        size = context_size or self.context_size

        # Reuse S-NIAH task with specified size
        sniah = SNIAHTask()
        instance = sniah.generate(seed=seed, context_size=size)

        # Update metadata
        instance.metadata = instance.metadata or {}
        instance.metadata["scaling_test"] = True
        instance.metadata["target_size"] = size

        return instance

    def check(self, response: str, expected: str) -> bool:
        # Same as S-NIAH
        code = expected.split("-", 1)[1] if "-" in expected else expected
        return code.upper() in response.upper()


@register_task("qa_retrieval")
class QARetrievalTask(BaseTask):
    """
    Question-Answer Retrieval

    Find specific facts scattered throughout a document.
    More realistic than arbitrary code retrieval.
    """

    description = "Answer questions from scattered facts in text"
    difficulty = "medium"

    def generate(self, seed: int = 42, context_size: int = 50000, num_facts: int = 5, **kwargs) -> TaskInstance:
        random.seed(seed)

        # Generate realistic fact-question pairs
        fact_templates = [
            ("The company was founded in {year}.", "What year was the company founded?", "{year}"),
            ("The CEO's name is {name}.", "Who is the CEO?", "{name}"),
            ("The headquarters is located in {city}.", "Where is the headquarters?", "{city}"),
            ("The annual revenue is ${amount} million.", "What is the annual revenue?", "${amount} million"),
            ("The company has {count} employees.", "How many employees does the company have?", "{count}"),
        ]

        values = [
            {"year": str(random.randint(1980, 2020))},
            {"name": random.choice(["John Smith", "Sarah Chen", "Michael Park", "Emma Wilson"])},
            {"city": random.choice(["New York", "San Francisco", "London", "Tokyo", "Berlin"])},
            {"amount": str(random.randint(100, 5000))},
            {"count": str(random.randint(100, 10000))},
        ]

        # Build context with facts
        noise = generate_noise(context_size, seed + 1)
        context = noise

        facts = []
        positions = sorted(
            random.sample(range(len(noise) // 10, 9 * len(noise) // 10), min(num_facts, len(fact_templates)))
        )

        offset = 0
        for i, pos in enumerate(positions):
            template, question, answer_tmpl = fact_templates[i]
            fact = template.format(**values[i])
            answer = answer_tmpl.format(**values[i])
            facts.append((question, answer))

            insert_text = f"\n\n{fact}\n\n"
            context = context[: pos + offset] + insert_text + context[pos + offset :]
            offset += len(insert_text)

        # Pick a random question
        question, expected = random.choice(facts)

        task = f"""Based on the text, answer this question:

{question}

Return ONLY the answer, nothing else."""

        return TaskInstance(task=task, context=context, expected=expected, metadata={"all_facts": facts})

    def check(self, response: str, expected: str) -> bool:
        # Flexible matching - expected value should appear in response
        expected_clean = expected.strip().lower()
        response_clean = response.strip().lower()
        return expected_clean in response_clean


# =============================================================================
# Task Factory
# =============================================================================


def get_task(name: str, **kwargs) -> BaseTask:
    """Get a task instance by name."""
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {name}. Available: {available}")
    return TASK_REGISTRY[name](**kwargs) if kwargs else TASK_REGISTRY[name]()


def list_tasks() -> list[str]:
    """List all registered tasks."""
    return list(TASK_REGISTRY.keys())
