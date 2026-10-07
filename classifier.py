"""
gestalt_extractor.classifier
============================
Multi-domain heuristic classifier for mathematical manuscripts.
Categorizes ParsedDocument instances into domain taxonomies using
multi-tier weighted keyword counting, LaTeX formula matching,
and section hierarchy analysis with deterministic fallback.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Sequence
from pathlib import Path

from gestalt_extractor.config import (
    DEFAULT_DOMAINS,
    DOMAIN_KEYWORDS as BASE_DOMAIN_KEYWORDS,
    ExtractorConfig,
)
from gestalt_extractor.parser import ParsedDocument

logger = logging.getLogger(__name__)

# Fallback domain identifier
DEFAULT_FALLBACK_DOMAIN: str = "General_Applied_Math"

# Default feature weights
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "title": 5.0,
    "section": 3.0,
    "math": 3.0,
    "theorem": 2.5,
    "algorithm": 2.0,
    "prose": 1.0,
}

# Enriched Domain Keywords with word-boundary awareness
EXTENDED_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "ML_Optimization": [
        r"\b(?:convex|non-convex|strongly\s+convex)\b",
        r"\b(?:gradient|gradients|gradient\s+descent|stochastic\s+gradient\s+descent|sgd)\b",
        r"\b(?:loss\s+function|loss\s+surface|empirical\s+risk|risk\s+minimization)\b",
        r"\b(?:learning\s+rate|learning\s+rates|lr\s+schedule)\b",
        r"\b(?:weight\s+decay|regularization|regularizer)\b",
        r"\b(?:adam|adamw|rmsprop|adagrad|sgd\s+momentum)\b",
        r"\b(?:grokfast|grokking|delayed\s+generalization)\b",
        r"\b(?:momentum|low-pass\s+filter(?:ing)?|exponential\s+moving\s+average|ema)\b",
        r"\b(?:backpropagation|backprop|autograd)\b",
        r"\b(?:eigenvalue|eigenvalues|singular\s+value|singular\s+values|svd)\b",
        r"\b(?:transformer|transformers|attention|self-attention|multi-head\s+attention)\b",
        r"\b(?:bits/parameter|bits\s+per\s+parameter)\b",
        r"\b(?:egalitarian\s+gradient\s+descent|egd)\b",
        r"\b(?:neural\s+network|neural\s+networks|deep\s+learning)\b",
        r"\b(?:overparameterized|overparameterization|generalization\s+bound)\b",
        r"\b(?:conditioning|ill-conditioned|condition\s+number|hessian)\b",
        r"\b(?:optimizer|optimizers|minibatch|epoch|epochs)\b",
    ],
    "Graph_Combinatorics": [
        r"\b(?:vertex|vertices)\b",
        r"\b(?:edge|edges)\b",
        r"\b(?:clique|cliques)\b",
        r"\b(?:network|networks)\b",
        r"\b(?:adjacency\s+matrix|adjacency\s+matrices|adjacency)\b",
        r"\b(?:directed\s+acyclic\s+graphs?|dags?)\b",
        r"\b(?:topological|topology|topological\s+sort)\b",
        r"\b(?:graph\s+isomorphism|isomorphism)\b",
        r"\b(?:bipartite|bipartite\s+graph)\b",
        r"\b(?:structural\s+causal\s+models?|scms?)\b",
        r"\b(?:causal\s+graph|causal\s+network|causal\s+topology|causal\s+chain)\b",
        r"\b(?:counterfactual|counterfactuals|do-calculus|judea\s+pearl)\b",
        r"\b(?:shortest\s+path|dijkstra|spanning\s+tree|planar\s+graph|chromatic\s+number)\b",
        r"\b(?:combinatorics|combinatorial|graph\s+theory)\b",
        r"\b(?:node|nodes)\b",
    ],
    "Crypto_Number_Theory": [
        r"\b(?:finite\s+field|finite\s+fields|galois\s+field)\b",
        r"\b(?:prime|primes|prime\s+number|prime\s+factorization)\b",
        r"\b(?:l-function|l-functions|dirichlet\s+l-function)\b",
        r"\b(?:zeta|riemann\s+zeta|zeta\s+function)\b",
        r"\b(?:elliptic\s+curve|elliptic\s+curves)\b",
        r"\b(?:quadratic\s+residue|quadratic\s+residues|residue)\b",
        r"\b(?:discrete\s+log|discrete\s+logarithm)\b",
        r"\b(?:modulus|modular\s+arithmetic|congruence)\b",
        r"\b(?:isogeny|isogenies)\b",
        r"\b(?:zero-knowledge|zk-snark|zk-stark|zksnark)\b",
        r"\b(?:cryptography|cryptographic|diffie-hellman|rsa|cipher)\b",
        r"\b(?:group\s+homomorphism|algebraic\s+number\s+theory)\b",
    ],
    "Fluid_Dynamics": [
        r"\b(?:navier-stokes|navier\s+stokes)\b",
        r"\b(?:fluid|fluids|fluid\s+mechanics|fluid\s+dynamics)\b",
        r"\b(?:viscosity|viscous|kinematic\s+viscosity)\b",
        r"\b(?:turbulence|turbulent\s+flow|laminar\s+flow)\b",
        r"\b(?:velocity\s+field|velocity\s+vector)\b",
        r"\b(?:reynolds\s+number)\b",
        r"\b(?:incompressible|compressible\s+flow)\b",
        r"\b(?:boundary\s+layer|boundary\s+layers)\b",
        r"\b(?:vorticity|vortex|vortices)\b",
        r"\b(?:continuity\s+equation|euler\s+equations)\b",
        r"\b(?:streamfunction|stream\s+function|bernoulli)\b",
    ],
    "General_Applied_Math": [
        r"\b(?:calculus|differential\s+equations?|ordinary\s+differential\s+equations?|ode|pde)\b",
        r"\b(?:linear\s+algebra|matrix\s+theory|vector\s+space)\b",
        r"\b(?:real\s+analysis|complex\s+analysis|functional\s+analysis)\b",
        r"\b(?:numerical\s+analysis|numerical\s+methods|finite\s+element)\b",
        r"\b(?:probability|statistics|random\s+variable|expectation|variance)\b",
        r"\b(?:hilbert\s+space|banach\s+space|metric\s+space)\b",
        r"\b(?:fourier\s+transform|laplace\s+transform)\b",
        r"\b(?:operator\s+theory|measure\s+theory)\b",
    ],
}

# Domain Formula & Notation Pattern Lexicon
DOMAIN_FORMULA_PATTERNS: Dict[str, List[str]] = {
    "ML_Optimization": [
        r"\\nabla(?!\s*(?:\\cdot|\\times|\^2|\^\{2\}))",
        r"\\mathcal\{L\}",
        r"\\mathcal\{D\}",
        r"\\theta",
        r"\\mathbf\{g\}",
        r"\\mathbf\{m\}",
        r"\\mathbf\{W\}",
        r"\\mathbf\{G\}",
        r"\\mathbf\{U\}",
        r"\\mathbf\{\\Sigma\}",
        r"\\mathbf\{V\}",
        r"\\sigma_[0-9i]",
        r"\\arg\\min",
        r"\\min_\\theta",
        r"\\mathbb\{E\}",
        r"\\operatorname\{rank\}|\\mathrm\{rank\}",
        r"\\operatorname\{Tr\}|\\mathrm\{Tr\}",
    ],
    "Graph_Combinatorics": [
        r"\\rightarrow|\\to",
        r"\\leftarrow|\\gets",
        r"\bdo\s*\(",
        r"\(V,\s*E\)|\(V,E\)",
        r"\\mathcal\{G\}",
        r"\\mathcal\{V\}",
        r"\\mathcal\{E\}",
        r"\\deg\s*\(",
        r"A_\{[a-z0-9,]+\}",
        r"\\binom\{",
        r"\\chi\(",
        r"\\omega\(",
        r"\\alpha\(",
    ],
    "Crypto_Number_Theory": [
        r"\\mathbb\{F\}",
        r"\\mathbb\{Z\}/",
        r"\\mathbb\{Z\}_",
        r"\\equiv",
        r"\\pmod",
        r"\\bmod",
        r"\\zeta\s*\(",
        r"L\s*\([s\d]",
        r"\\gcd",
        r"\\operatorname\{lcm\}",
        r"g\^[a-z0-9]",
        r"E\(\\mathbb\{F\}",
    ],
    "Fluid_Dynamics": [
        r"\\nabla\s*\\cdot",
        r"\\nabla\s*\\times",
        r"\\nabla\^2",
        r"\\Delta",
        r"\\frac\{\\partial\s*\\mathbf\{u\}\}\{\\partial\s*t\}",
        r"\\partial_t\s*\\mathbf\{u\}",
        r"\\mathrm\{Re\}|Re\s*=",
        r"\\rho",
        r"\\nu",
        r"\\mu",
    ],
    "General_Applied_Math": [
        r"\\int",
        r"\\sum",
        r"\\prod",
        r"\\frac\{d\}\{dx\}",
        r"\\lim_",
        r"\\epsilon",
        r"\\delta",
    ],
}


class DomainClassifier:
    """Multi-domain manuscript classifier adhering to Milestone 2 requirements."""

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        custom_keywords: Optional[Dict[str, List[str]]] = None,
        custom_formulas: Optional[Dict[str, List[str]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config or ExtractorConfig()
        self.domains: Tuple[str, ...] = DEFAULT_DOMAINS
        self.weights = dict(DEFAULT_FEATURE_WEIGHTS)
        if weights:
            self.weights.update(weights)

        # Merge base keywords with extended and custom keywords
        self.keyword_patterns: Dict[str, List[re.Pattern]] = {}
        source_keywords: Dict[str, List[str]] = {d: list(kws) for d, kws in EXTENDED_DOMAIN_KEYWORDS.items()}
        if custom_keywords:
            for domain, kws in custom_keywords.items():
                if domain in source_keywords:
                    source_keywords[domain].extend(kws)
                else:
                    source_keywords[domain] = list(kws)

        # Precompile keyword regexes
        for domain in self.domains:
            patterns = []
            for kw in source_keywords.get(domain, []):
                try:
                    patterns.append(re.compile(kw, re.IGNORECASE))
                except re.error as e:
                    logger.warning("Invalid regex for domain %s (%s): %s", domain, kw, e)
            self.keyword_patterns[domain] = patterns

        # Merge and precompile formula regexes
        self.formula_patterns: Dict[str, List[re.Pattern]] = {}
        source_formulas: Dict[str, List[str]] = {d: list(fms) for d, fms in DOMAIN_FORMULA_PATTERNS.items()}
        if custom_formulas:
            for domain, fms in custom_formulas.items():
                if domain in source_formulas:
                    source_formulas[domain].extend(fms)
                else:
                    source_formulas[domain] = list(fms)

        for domain in self.domains:
            patterns = []
            for fm in source_formulas.get(domain, []):
                try:
                    patterns.append(re.compile(fm, re.IGNORECASE))
                except re.error as e:
                    logger.warning("Invalid formula pattern for domain %s (%s): %s", domain, fm, e)
            self.formula_patterns[domain] = patterns

    def classify(self, doc: ParsedDocument) -> str:
        """Classifies a ParsedDocument and returns the assigned domain label."""
        domain, _ = self.classify_with_scores(doc)
        return domain

    def classify_with_scores(
        self, doc: ParsedDocument
    ) -> Tuple[str, Dict[str, float]]:
        """Classifies a ParsedDocument and returns the domain label and score breakdown."""
        if doc is None:
            return DEFAULT_FALLBACK_DOMAIN, {d: 0.0 for d in self.domains}

        # 1. Feature Extraction
        title_text = doc.title or ""
        section_titles = " \n ".join(
            sec.get("title", "") for sec in (doc.sections or []) if isinstance(sec, dict)
        )
        prose_text = doc.clean_text or ""
        math_text = " \n ".join(doc.math_blocks or [])
        theorem_text = " \n ".join(doc.theorems or [])
        algo_text = " \n ".join(doc.algorithms or [])

        scores: Dict[str, float] = {d: 0.0 for d in self.domains}

        # 2. Multi-Pass Feature Scoring
        for domain in self.domains:
            kw_regexes = self.keyword_patterns.get(domain, [])
            fm_regexes = self.formula_patterns.get(domain, [])

            # Pass A: Title matches (Weight: 5.0)
            title_hits = sum(len(rgx.findall(title_text)) for rgx in kw_regexes)
            scores[domain] += title_hits * self.weights["title"]

            # Pass B: Section header matches (Weight: 3.0)
            sec_hits = sum(len(rgx.findall(section_titles)) for rgx in kw_regexes)
            scores[domain] += sec_hits * self.weights["section"]

            # Pass C: Prose text matches (Weight: 1.0)
            prose_hits = sum(len(rgx.findall(prose_text)) for rgx in kw_regexes)
            scores[domain] += prose_hits * self.weights["prose"]

            # Pass D: Math block formula matches (Weight: 3.0)
            math_hits = sum(len(rgx.findall(math_text)) for rgx in fm_regexes)
            # Math blocks can also contain domain text keywords
            math_kw_hits = sum(len(rgx.findall(math_text)) for rgx in kw_regexes)
            scores[domain] += (math_hits + math_kw_hits) * self.weights["math"]

            # Pass E: Theorem environment matches (Weight: 2.5)
            thm_hits = sum(len(rgx.findall(theorem_text)) for rgx in kw_regexes)
            thm_fm_hits = sum(len(rgx.findall(theorem_text)) for rgx in fm_regexes)
            scores[domain] += (thm_hits + thm_fm_hits) * self.weights["theorem"]

            # Pass F: Algorithm block matches (Weight: 2.0)
            algo_hits = sum(len(rgx.findall(algo_text)) for rgx in kw_regexes)
            scores[domain] += algo_hits * self.weights["algorithm"]

            # Round score to 4 decimal places for consistency
            scores[domain] = round(scores[domain], 4)

        # 3. Decision Matrix & Fallback Logic
        max_score = max(scores.values())

        # Fallback Rule 1: Zero score fallback
        if max_score <= 0.0:
            logger.debug(
                "Document '%s' scored 0 across all domains. Falling back to %s.",
                getattr(doc, "title", "Untitled"),
                DEFAULT_FALLBACK_DOMAIN,
            )
            return DEFAULT_FALLBACK_DOMAIN, scores

        # Fallback Rule 2: Tie-breaker fallback
        leaders = [d for d, s in scores.items() if s == max_score]
        if len(leaders) > 1:
            logger.debug(
                "Document '%s' had a tie between %s with score %.2f. Falling back to %s.",
                getattr(doc, "title", "Untitled"),
                leaders,
                max_score,
                DEFAULT_FALLBACK_DOMAIN,
            )
            return DEFAULT_FALLBACK_DOMAIN, scores

        # Decisive winner
        winner = leaders[0]
        return winner, scores


# Backward-compatibility alias
Classifier = DomainClassifier


def classify_document(doc: ParsedDocument) -> str:
    """Convenience helper to classify a document using default settings."""
    classifier = DomainClassifier()
    return classifier.classify(doc)


def classify_document_with_scores(
    doc: ParsedDocument,
) -> Tuple[str, Dict[str, float]]:
    """Convenience helper to classify a document with scores using default settings."""
    classifier = DomainClassifier()
    return classifier.classify_with_scores(doc)


__all__ = [
    "DomainClassifier",
    "Classifier",
    "classify_document",
    "classify_document_with_scores",
    "DEFAULT_FALLBACK_DOMAIN",
    "DEFAULT_FEATURE_WEIGHTS",
    "EXTENDED_DOMAIN_KEYWORDS",
    "DOMAIN_FORMULA_PATTERNS",
]
