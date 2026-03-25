"""
LLM-guided search agent for the AutoResearch loop.

Uses an LLM to analyze experiment history and propose intelligent
next steps, rather than random search.
"""
import os
import json
import random
from typing import Optional, Tuple

from .agent import Agent, Proposal

# Model imports - same pool as RandomSearchAgent
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDRegressor, SGDClassifier, BayesianRidge,
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    BaggingRegressor, BaggingClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, QuantileTransformer, PowerTransformer,
)
from sklearn.decomposition import PCA

# Optional boosting libraries
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# All available models the LLM can choose from
MODEL_REGISTRY = {
    "regression": {
        "LinearRegression": lambda **kw: LinearRegression(),
        "Ridge": lambda **kw: Ridge(alpha=kw.get("alpha", 1.0), random_state=42),
        "Lasso": lambda **kw: Lasso(alpha=kw.get("alpha", 0.1), random_state=42, max_iter=2000),
        "ElasticNet": lambda **kw: ElasticNet(
            alpha=kw.get("alpha", 0.1), l1_ratio=kw.get("l1_ratio", 0.5),
            random_state=42, max_iter=2000
        ),
        "BayesianRidge": lambda **kw: BayesianRidge(),
        "SGD": lambda **kw: SGDRegressor(
            alpha=kw.get("alpha", 0.0001),
            learning_rate=kw.get("learning_rate", "invscaling"),
            random_state=42, max_iter=1000,
        ),
        "RandomForest": lambda **kw: RandomForestRegressor(
            n_estimators=kw.get("n_estimators", 100),
            max_depth=kw.get("max_depth", None),
            min_samples_split=kw.get("min_samples_split", 2),
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": lambda **kw: GradientBoostingRegressor(
            n_estimators=kw.get("n_estimators", 100),
            learning_rate=kw.get("learning_rate", 0.1),
            max_depth=kw.get("max_depth", 3),
            subsample=kw.get("subsample", 1.0),
            random_state=42,
        ),
        "HistGradientBoosting": lambda **kw: HistGradientBoostingRegressor(
            max_iter=kw.get("max_iter", 100),
            learning_rate=kw.get("learning_rate", 0.1),
            max_depth=kw.get("max_depth", None),
            min_samples_leaf=kw.get("min_samples_leaf", 20),
            random_state=42,
        ),
        "ExtraTrees": lambda **kw: ExtraTreesRegressor(
            n_estimators=kw.get("n_estimators", 100),
            max_depth=kw.get("max_depth", None),
            random_state=42, n_jobs=-1,
        ),
        "AdaBoost": lambda **kw: AdaBoostRegressor(
            n_estimators=kw.get("n_estimators", 50),
            learning_rate=kw.get("learning_rate", 1.0),
            random_state=42,
        ),
        "KNN": lambda **kw: KNeighborsRegressor(
            n_neighbors=kw.get("n_neighbors", 5),
            weights=kw.get("weights", "uniform"),
            n_jobs=-1,
        ),
        "SVR": lambda **kw: SVR(
            C=kw.get("C", 1.0), epsilon=kw.get("epsilon", 0.1),
            kernel=kw.get("kernel", "rbf"),
        ),
        "DecisionTree": lambda **kw: DecisionTreeRegressor(
            max_depth=kw.get("max_depth", None),
            min_samples_split=kw.get("min_samples_split", 2),
            random_state=42,
        ),
        **({
            "XGBoost": lambda **kw: XGBRegressor(
                n_estimators=kw.get("n_estimators", 200),
                learning_rate=kw.get("learning_rate", 0.1),
                max_depth=kw.get("max_depth", 5),
                subsample=kw.get("subsample", 0.8),
                colsample_bytree=kw.get("colsample_bytree", 0.8),
                reg_alpha=kw.get("reg_alpha", 0),
                reg_lambda=kw.get("reg_lambda", 1),
                random_state=42, n_jobs=-1, verbosity=0,
            ),
        } if _HAS_XGB else {}),
        **({
            "LightGBM": lambda **kw: LGBMRegressor(
                n_estimators=kw.get("n_estimators", 200),
                learning_rate=kw.get("learning_rate", 0.1),
                max_depth=kw.get("max_depth", -1),
                num_leaves=kw.get("num_leaves", 31),
                subsample=kw.get("subsample", 0.8),
                colsample_bytree=kw.get("colsample_bytree", 0.8),
                reg_alpha=kw.get("reg_alpha", 0),
                reg_lambda=kw.get("reg_lambda", 0),
                random_state=42, n_jobs=-1, verbose=-1,
            ),
        } if _HAS_LGBM else {}),
    },
    "classification": {
        "LogisticRegression": lambda **kw: LogisticRegression(
            C=kw.get("C", 1.0), max_iter=1000, random_state=42,
        ),
        "SGD": lambda **kw: SGDClassifier(
            alpha=kw.get("alpha", 0.0001),
            loss=kw.get("loss", "hinge"),
            random_state=42, max_iter=1000,
        ),
        "RandomForest": lambda **kw: RandomForestClassifier(
            n_estimators=kw.get("n_estimators", 100),
            max_depth=kw.get("max_depth", None),
            min_samples_split=kw.get("min_samples_split", 2),
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": lambda **kw: GradientBoostingClassifier(
            n_estimators=kw.get("n_estimators", 100),
            learning_rate=kw.get("learning_rate", 0.1),
            max_depth=kw.get("max_depth", 3),
            subsample=kw.get("subsample", 1.0),
            random_state=42,
        ),
        "HistGradientBoosting": lambda **kw: HistGradientBoostingClassifier(
            max_iter=kw.get("max_iter", 100),
            learning_rate=kw.get("learning_rate", 0.1),
            max_depth=kw.get("max_depth", None),
            min_samples_leaf=kw.get("min_samples_leaf", 20),
            random_state=42,
        ),
        "ExtraTrees": lambda **kw: ExtraTreesClassifier(
            n_estimators=kw.get("n_estimators", 100),
            max_depth=kw.get("max_depth", None),
            random_state=42, n_jobs=-1,
        ),
        "AdaBoost": lambda **kw: AdaBoostClassifier(
            n_estimators=kw.get("n_estimators", 50),
            learning_rate=kw.get("learning_rate", 1.0),
            random_state=42,
        ),
        "KNN": lambda **kw: KNeighborsClassifier(
            n_neighbors=kw.get("n_neighbors", 5),
            weights=kw.get("weights", "uniform"),
            n_jobs=-1,
        ),
        "SVC": lambda **kw: SVC(
            C=kw.get("C", 1.0), kernel=kw.get("kernel", "rbf"),
            random_state=42,
        ),
        "DecisionTree": lambda **kw: DecisionTreeClassifier(
            max_depth=kw.get("max_depth", None),
            min_samples_split=kw.get("min_samples_split", 2),
            random_state=42,
        ),
        **({
            "XGBoost": lambda **kw: XGBClassifier(
                n_estimators=kw.get("n_estimators", 200),
                learning_rate=kw.get("learning_rate", 0.1),
                max_depth=kw.get("max_depth", 5),
                random_state=42, n_jobs=-1, verbosity=0,
                use_label_encoder=False, eval_metric='logloss',
            ),
        } if _HAS_XGB else {}),
        **({
            "LightGBM": lambda **kw: LGBMClassifier(
                n_estimators=kw.get("n_estimators", 200),
                learning_rate=kw.get("learning_rate", 0.1),
                num_leaves=kw.get("num_leaves", 31),
                random_state=42, n_jobs=-1, verbose=-1,
            ),
        } if _HAS_LGBM else {}),
    },
}

PREPROCESSOR_REGISTRY = {
    "None": lambda **kw: None,
    "StandardScaler": lambda **kw: StandardScaler(),
    "MinMaxScaler": lambda **kw: MinMaxScaler(),
    "RobustScaler": lambda **kw: RobustScaler(),
    "QuantileTransformer": lambda **kw: QuantileTransformer(
        n_quantiles=kw.get("n_quantiles", 100), random_state=42,
    ),
    "PowerTransformer": lambda **kw: PowerTransformer(
        method=kw.get("method", "yeo-johnson"), standardize=True,
    ),
    "PCA": lambda **kw: PCA(
        n_components=kw.get("n_components", 0.95), random_state=42,
    ),
    "PolynomialFeatures": lambda **kw: PolynomialFeatures(
        degree=kw.get("degree", 2), include_bias=False,
    ),
}


def _build_prompt(state: dict) -> str:
    """Build a prompt for the LLM based on experiment history."""
    task = state.get("task_type", "regression")
    best_score = state.get("best_score")
    best_model = state.get("best_model", "unknown")
    iteration = state.get("iteration", 0)
    metric_direction = state.get("metric_direction", "minimize")
    history = state.get("history", [])
    data_summary = state.get("data_summary", "")

    # Format recent history (last 10 entries)
    history_text = ""
    if history:
        recent = history[-10:]
        history_text = "\n".join(
            f"  - {h['description']}: {h['score']:.4f} {'✓ improvement' if h.get('improved') else ''}"
            for h in recent
        )
    else:
        history_text = "  (no history yet — first iteration)"

    models_available = ", ".join(MODEL_REGISTRY[task].keys())
    preprocessors_available = ", ".join(PREPROCESSOR_REGISTRY.keys())

    return f"""You are an AutoML research agent optimizing a prediction pipeline on tabular data.

Task: {task} | Metric: {metric_direction} | Best so far: {best_model} = {best_score:.6f}
Iteration: {iteration} | {f'Data: {data_summary}' if data_summary else ''}

Recent experiments:
{history_text}

STRATEGY RULES:
- For tabular data, tree-based models (GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, RandomForest) almost always win.
- DO NOT try SVR, KNN, or LinearRegression when tree-based models are clearly ahead.
- Focus on HYPERPARAMETER TUNING around the current best model type.
- Try small variations: slightly different learning_rate, max_depth, n_estimators, subsample, colsample_bytree.
- If GBM-family is winning, try: more estimators with lower learning rate, different depth, subsample<1.0.
- XGBoost and LightGBM with reg_alpha/reg_lambda regularization can beat vanilla GBM.
- Preprocessing is usually NOT needed for tree models. Only use it for linear models.

Available models: {models_available}
Available preprocessors: {preprocessors_available}

Respond with ONLY a JSON object:
{{
  "model": "<model_name>",
  "model_params": {{"param1": value1}},
  "preprocessor": "None",
  "preprocessor_params": {{}},
  "rationale": "<why this specific config>"
}}"""


def _parse_llm_response(response_text: str, task_type: str) -> Optional[Proposal]:
    """Parse LLM JSON response into a Proposal."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    model_name = data.get("model", "")
    model_params = data.get("model_params", {})
    preprocessor_name = data.get("preprocessor", "None")
    preprocessor_params = data.get("preprocessor_params", {})
    rationale = data.get("rationale", "LLM suggestion")

    # Build model
    registry = MODEL_REGISTRY.get(task_type, {})
    model_factory = registry.get(model_name)
    if not model_factory:
        return None

    try:
        # Sanitize params: convert to correct types
        clean_params = {}
        for k, v in model_params.items():
            if v == "null" or v == "None":
                clean_params[k] = None
            else:
                clean_params[k] = v
        model = model_factory(**clean_params)
    except Exception:
        return None

    # Build preprocessor
    preprocessor = None
    prep_desc = ""
    if preprocessor_name and preprocessor_name != "None":
        prep_factory = PREPROCESSOR_REGISTRY.get(preprocessor_name)
        if prep_factory:
            try:
                preprocessor = prep_factory(**preprocessor_params)
                prep_desc = preprocessor_name
            except Exception:
                pass

    description = f"{prep_desc} + {model_name}({model_params})" if prep_desc else f"{model_name}({model_params})"

    return Proposal(
        description=description,
        rationale=rationale,
        model=model,
        preprocessor=preprocessor,
    )


class LLMSearchAgent(Agent):
    """
    Agent that uses an LLM to analyze experiment history and propose
    intelligent next steps.
    
    Falls back to RandomSearchAgent on LLM failures.
    """

    def __init__(self, task_type: str, provider: str = "groq", model: str = None):
        self.task_type = task_type
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.history = []
        self.llm_failures = 0
        self.total_calls = 0
        self._client = None
        
        # Fallback agent
        from .agent import RandomSearchAgent
        self._fallback = RandomSearchAgent(task_type)

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "groq":
            return "llama-3.3-70b-versatile"
        elif provider == "openai":
            return "gpt-4o-mini"
        return "llama-3.3-70b-versatile"

    def _get_client(self):
        if self._client is not None:
            return self._client
        
        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM and return text response."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    def propose(self, state: dict) -> Proposal:
        """
        Use LLM to propose a pipeline, with fallback to random.
        
        Strategy: LLM every 3 iterations, random in between.
        This balances exploration speed (random is instant) with
        intelligent guidance (LLM analyzes patterns).
        """
        iteration = state.get("iteration", 0)
        
        # Track history
        if "last_proposal" in state and state.get("last_score") is not None:
            self.history.append({
                "description": state["last_proposal"],
                "score": state["last_score"],
                "improved": state.get("last_improved", False),
            })

        # Use LLM every 3rd iteration (saves API calls, still guides search)
        use_llm = (iteration % 3 == 0) and iteration > 0
        
        if use_llm:
            self.total_calls += 1
            enriched_state = {**state, "history": self.history}
            prompt = _build_prompt(enriched_state)
            response = self._call_llm(prompt)
            
            if response:
                proposal = _parse_llm_response(response, self.task_type)
                if proposal:
                    return proposal
            
            self.llm_failures += 1
        
        # Fallback to random search
        return self._fallback.propose(state)

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return 1.0 - (self.llm_failures / self.total_calls)
