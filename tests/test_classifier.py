"""
Unit tests for gestalt_extractor.classifier (Domain Classifier)
==============================================================
Verifies Milestone 2 Acceptance Criteria:
- Standardized taxonomy: ML_Optimization, Graph_Combinatorics, Crypto_Number_Theory, Fluid_Dynamics, General_Applied_Math.
- Weighted feature scoring (title 5.0, section 3.0, math 3.0, theorem 2.5, algorithm 2.0, prose 1.0).
- Substring boundary defenses (knowledge != edge, dagger != dag).
- Deterministic fallback on 0-score or tie.
- Real testbed paper verification: memo.md, TECHNICAL-NOTES.md, essay_3_gettier.md, sample_bounds.*.
"""

import unittest
from pathlib import Path

from gestalt_extractor.config import DEFAULT_DOMAINS
from gestalt_extractor.parser import DocumentParser, ParsedDocument
from gestalt_extractor.classifier import (
    DomainClassifier,
    Classifier,
    classify_document,
    classify_document_with_scores,
    DEFAULT_FALLBACK_DOMAIN,
)


class TestClassifierTaxonomyAndContract(unittest.TestCase):
    """Verifies domain taxonomy completeness and interface signatures."""

    def setUp(self):
        self.classifier = DomainClassifier()
        self.parser = DocumentParser()

    def test_domain_taxonomy_completeness(self):
        expected = {
            "ML_Optimization",
            "Graph_Combinatorics",
            "Crypto_Number_Theory",
            "Fluid_Dynamics",
            "General_Applied_Math",
        }
        self.assertEqual(set(self.classifier.domains), expected)
        self.assertEqual(set(DEFAULT_DOMAINS), expected)

    def test_interface_signatures_and_types(self):
        doc = self.parser.parse_text("# Test Title\nSome content.")
        domain = self.classifier.classify(doc)
        domain_with_scores, scores = self.classifier.classify_with_scores(doc)

        self.assertIsInstance(domain, str)
        self.assertIsInstance(domain_with_scores, str)
        self.assertEqual(domain, domain_with_scores)
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 5)
        for d in DEFAULT_DOMAINS:
            self.assertIn(d, scores)
            self.assertIsInstance(scores[d], float)

    def test_none_document_handling(self):
        domain, scores = self.classifier.classify_with_scores(None)
        self.assertEqual(domain, DEFAULT_FALLBACK_DOMAIN)
        self.assertTrue(all(s == 0.0 for s in scores.values()))

    def test_convenience_functions(self):
        doc = self.parser.parse_text("# Stochastic Gradient Descent\nConvex optimization of loss function.")
        domain = classify_document(doc)
        self.assertEqual(domain, "ML_Optimization")

        dom2, scores = classify_document_with_scores(doc)
        self.assertEqual(dom2, "ML_Optimization")
        self.assertGreater(scores["ML_Optimization"], 0.0)

    def test_backward_compatibility_alias(self):
        self.assertIs(Classifier, DomainClassifier)


class TestClassifierScoringAndDefenses(unittest.TestCase):
    """Verifies word-boundary defenses, multi-tier weights, and fallback logic."""

    def setUp(self):
        self.classifier = DomainClassifier()
        self.parser = DocumentParser()

    def test_word_boundary_safety_knowledge_vs_edge(self):
        # "knowledge", "acknowledged", "sledge" must NOT trigger "edge" in Graph_Combinatorics
        text = "This paper analyzes human knowledge, acknowledged evidence, and sledge hammer techniques."
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["Graph_Combinatorics"], 0.0)

    def test_word_boundary_safety_dagger_vs_dag(self):
        # "dagger" must NOT trigger "dag" in Graph_Combinatorics
        text = "The dagger was placed beside the table."
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["Graph_Combinatorics"], 0.0)

    def test_zero_score_fallback(self):
        doc = self.parser.parse_text("The quick brown fox jumps over the lazy dog.")
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, DEFAULT_FALLBACK_DOMAIN)
        self.assertTrue(all(s == 0.0 for s in scores.values()))

    def test_tie_breaking_fallback(self):
        # Exactly 1 match for ML (convex) and 1 match for Crypto (prime)
        doc = ParsedDocument(
            path=Path("tie.md"),
            title="Tie Document",
            raw_text="A convex set and a prime number.",
            clean_text="A convex set and a prime number.",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["ML_Optimization"], scores["Crypto_Number_Theory"])
        self.assertGreater(scores["ML_Optimization"], 0.0)
        # Should detect tie and fall back to General_Applied_Math
        self.assertEqual(domain, DEFAULT_FALLBACK_DOMAIN)

    def test_title_weight_dominance(self):
        # Title keyword has weight 5.0; prose keyword has weight 1.0
        doc = ParsedDocument(
            path=Path("weighted.md"),
            title="Navier-Stokes Equations",
            raw_text="We discuss convex functions.",
            clean_text="We discuss convex functions.",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["Fluid_Dynamics"], 5.0)
        self.assertEqual(scores["ML_Optimization"], 1.0)
        self.assertEqual(domain, "Fluid_Dynamics")

    def test_math_block_scoring(self):
        doc = ParsedDocument(
            path=Path("math_test.md"),
            title="Mathematical Test",
            raw_text=r"$$\nabla \mathcal{L}(\theta) = 0$$",
            clean_text="",
            math_blocks=[r"\nabla \mathcal{L}(\theta) = 0"],
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "ML_Optimization")
        self.assertGreater(scores["ML_Optimization"], 0.0)

    def test_classify_with_scores_sections_none_does_not_crash(self):
        """Remediation Verification: ParsedDocument(sections=None) must not raise TypeError."""
        doc = ParsedDocument(
            path=Path("no_sections.md"),
            title="A Treatise Without Sections",
            raw_text="Sample text on convex functions.",
            clean_text="Sample text on convex functions.",
            sections=None,  # Explicitly None to test defensive guard
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertIsInstance(domain, str)
        self.assertIsInstance(scores, dict)
        self.assertEqual(domain, "ML_Optimization")

    def test_fluid_divergence_and_curl_not_scored_as_ml(self):
        """Remediation Verification: \nabla \cdot and \nabla \times must not award ML points."""
        doc = ParsedDocument(
            path=Path("pure_divergence.tex"),
            title="Divergence-Free Vector Field",
            raw_text=r"$$\nabla \cdot \mathbf{u} = 0, \quad \nabla \times \mathbf{u} = \boldsymbol{\omega}$$",
            clean_text="",
            math_blocks=[
                r"\nabla \cdot \mathbf{u} = 0",
                r"\nabla \times \mathbf{u} = \boldsymbol{\omega}",
                r"\nabla^2 \mathbf{u} = 0",
                r"\nabla^{2} \phi = 0",
            ],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        # ML_Optimization must receive exactly 0 points from divergence/curl/Laplacian
        self.assertEqual(
            scores["ML_Optimization"],
            0.0,
            f"Regression: ML_Optimization received points ({scores['ML_Optimization']}) for fluid operators!",
        )
        self.assertGreater(scores["Fluid_Dynamics"], 0.0)
        self.assertEqual(domain, "Fluid_Dynamics")

    def test_ml_gradients_still_matched_by_ml_optimization(self):
        """Remediation Verification: Legitimate ML gradients remain matched."""
        doc = ParsedDocument(
            path=Path("ml_gradients.tex"),
            title="Optimization Dynamics",
            raw_text=r"$$\nabla \mathcal{L}(\theta) = \mathbf{g}_t, \quad \nabla_\theta f(x) = 0$$",
            clean_text="",
            math_blocks=[
                r"\nabla \mathcal{L}(\theta) = \mathbf{g}_t",
                r"\nabla_\theta f(x) = 0",
                r"\nabla_{\mathbf{w}} J(\mathbf{w}) = 0",
            ],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertGreater(scores["ML_Optimization"], 0.0)
        self.assertEqual(domain, "ML_Optimization")


class TestSyntheticDomains(unittest.TestCase):
    """Verifies decisive classification across all 5 domain categories using synthetic texts."""

    def setUp(self):
        self.classifier = DomainClassifier()
        self.parser = DocumentParser()

    def test_synthetic_ml_optimization(self):
        text = """# Empirical Risk Minimization and AdamW
We analyze stochastic gradient descent and grokfast momentum dynamics.
$$\nabla \mathcal{L}(\theta) = \mathbf{g}_t$$
The learning rate and weight decay control delayed generalization.
"""
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "ML_Optimization")

    def test_synthetic_graph_combinatorics(self):
        text = """# Structural Causal Models and Directed Acyclic Graphs
Let G = (V, E) be a directed acyclic graph representing causal topology.
We define the adjacency matrix and topological sort of vertices and edges.
Applying do-calculus and counterfactual interventions:
$$\text{do}(X = x)$$
"""
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "Graph_Combinatorics")

    def test_synthetic_crypto_number_theory(self):
        text = """# Elliptic Curves and Finite Fields
We study discrete logarithm hardness over elliptic curves in finite fields $\mathbb{F}_p$.
Zero-knowledge proofs and modular arithmetic for quadratic residues:
$$\mathbb{F}_q, \quad a \equiv b \pmod{p}$$
"""
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "Crypto_Number_Theory")

    def test_synthetic_fluid_dynamics(self):
        text = """# Incompressible Navier-Stokes Equations
We compute the velocity field and kinematic viscosity for high Reynolds number turbulent flow.
$$\nabla \cdot \mathbf{u} = 0, \quad \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$
The boundary layer vorticity dissipates kinetic energy.
"""
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "Fluid_Dynamics")

    def test_synthetic_general_applied_math(self):
        text = """# Hilbert Spaces and Functional Analysis
We investigate ordinary differential equations and boundary value problems in Banach space.
$$\int_0^\infty f(x) dx < \infty$$
Linear algebra and metric space completeness under Fourier transforms.
"""
        doc = self.parser.parse_text(text)
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "General_Applied_Math")


class TestRealWorkspaceDocumentsAndFixtures(unittest.TestCase):
    """Verifies classification accuracy on real workspace manuscripts and testbed fixtures."""

    def setUp(self):
        self.classifier = DomainClassifier()
        self.parser = DocumentParser()
        self.workspace_root = Path(__file__).resolve().parent.parent.parent

    def test_sample_bounds_fixtures_ml_optimization(self):
        fixtures_dir = self.workspace_root / "gestalt_extractor/testbed/fixtures"
        for fname in ["sample_bounds.md", "sample_bounds.tex"]:
            fpath = fixtures_dir / fname
            if fpath.exists():
                doc = self.parser.parse_file(fpath)
                self.assertIsNotNone(doc)
                domain = self.classifier.classify(doc)
                self.assertEqual(domain, "ML_Optimization")

    def test_real_workspace_memo_md(self):
        memo_path = self.workspace_root / "research/2026-09-18/grokking-swarm-autoresearch/grokking-notes/memo.md"
        if memo_path.exists():
            doc = self.parser.parse_file(memo_path)
            self.assertIsNotNone(doc)
            domain, scores = self.classifier.classify_with_scores(doc)
            self.assertEqual(domain, "ML_Optimization")
            self.assertGreater(scores["ML_Optimization"], scores["Graph_Combinatorics"])

    def test_real_workspace_technical_notes_md(self):
        notes_path = self.workspace_root / "research/2026-09-18/grokking-swarm-autoresearch/score-autoresearch-notes/TECHNICAL-NOTES.md"
        if notes_path.exists():
            doc = self.parser.parse_file(notes_path)
            self.assertIsNotNone(doc)
            domain, scores = self.classifier.classify_with_scores(doc)
            self.assertEqual(domain, "ML_Optimization")

    def test_real_workspace_gettier_essay_md(self):
        gettier_path = self.workspace_root / "sweetbrier_repo/competition/essay_3_gettier.md"
        if gettier_path.exists():
            doc = self.parser.parse_file(gettier_path)
            self.assertIsNotNone(doc)
            domain, scores = self.classifier.classify_with_scores(doc)
            self.assertEqual(domain, "Graph_Combinatorics")
            self.assertGreater(scores["Graph_Combinatorics"], scores["ML_Optimization"])


if __name__ == "__main__":
    unittest.main()
