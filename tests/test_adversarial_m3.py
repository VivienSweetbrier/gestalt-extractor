"""
gestalt_extractor.tests.test_adversarial_m3
===========================================
Milestone 3 Adversarial Challenge & Stress Test Suite:
1. Concurrency Stress & Determinism:
   - Validates that execution across workers=1, workers=4, workers=8 produces
     identical deterministic output records.
   - Validates high-worker concurrency (workers=16, 32).
   - Tests file ordering determinism across subdirectories.
2. Fault Tolerance & Thread-Safe Error Containment:
   - Corrupt, missing, permission-denied, or crashing files do NOT crash the batch.
   - Succeeded records are preserved; failed records are recorded in metadata["errors"].
   - 100% failure batches gracefully produce empty valid reports.
3. Exclusion Integrity:
   - Deep nested paths under node_modules, .agents, .git, build, dist are strictly omitted.
   - Custom dynamic exclusion patterns work cleanly.
4. Schema Enforcement & Serialization:
   - Disk JSON strictly adheres to the 4-key schema: file, domain, bounds, summary.
   - ExtractedRecord.from_dict strictly rejects missing or None required fields.
   - Concurrent atomic saves to the same destination path do not collide or corrupt.
"""

import os
import json
import uuid
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.extractor import GestaltExtractor, extract_file, extract_text
from gestalt_extractor.pipeline import ExtractionPipeline, run_extraction
from gestalt_extractor.models import (
    ExtractedRecord,
    ExtractedBound,
    ExtractionReport,
    SchemaValidationError,
)


class TestConcurrencyDeterminism(unittest.TestCase):
    """Hostile concurrency and determinism stress testing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Generate a diverse synthetic corpus of 12 mathematical manuscripts
        self.corpus_files = []
        domains_and_formulas = [
            ("ml_opt_1", "ML_Optimization", r"\|\nabla \mathcal{L}(\theta)\| \le \epsilon", "Grokfast gradient filter"),
            ("ml_opt_2", "ML_Optimization", r"\eta_t = \eta_0 / \sqrt{t}", "Learning rate decay schedule"),
            ("graph_1", "Graph_Combinatorics", r"\chi(G) \le \Delta(G) + 1", "Brooks chromatic graph bound"),
            ("graph_2", "Graph_Combinatorics", r"|E| \le 3|V| - 6", "Planar graph edge upper bound"),
            ("crypto_1", "Crypto_Number_Theory", r"|E(\mathbb{F}_p)| \le p + 1 + 2\sqrt{p}", "Hasse elliptic curve bound"),
            ("crypto_2", "Crypto_Number_Theory", r"g^x \equiv y \pmod{p}", "Discrete logarithm formulation"),
            ("fluid_1", "Fluid_Dynamics", r"\nabla \cdot \mathbf{u} = 0", "Incompressible Navier-Stokes continuity"),
            ("fluid_2", "Fluid_Dynamics", r"\mathrm{Re} = \frac{\rho u L}{\mu}", "Reynolds dimensionless number"),
            ("math_1", "General_Applied_Math", r"\|Ax - b\|_2 \le \delta", "Least squares residual bound"),
            ("math_2", "General_Applied_Math", r"\det(A) = \prod \lambda_i", "Matrix determinant eigenvalue identity"),
            ("complex_svd", "ML_Optimization", r"\sigma_{\max}(A) / \sigma_{\min}(A) \le \kappa", "Condition number bound"),
            ("ema_bound", "ML_Optimization", r"\alpha \in [0.9, 0.999]", "Exponential moving average factor"),
        ]

        for i, (name, domain_hint, formula, desc) in enumerate(domains_and_formulas):
            ext = ".md" if i % 2 == 0 else ".tex"
            p = self.root / f"paper_{i:02d}_{name}{ext}"
            if ext == ".md":
                content = (
                    f"# Paper {i}: {desc}\n\n"
                    f"Mathematical derivation for {domain_hint}.\n\n"
                    f"$$\n{formula}\n$$\n\n"
                    f"**Theorem {i}**: For all iterations $t \\ge 1$, the bound holds.\n"
                )
            else:
                content = (
                    r"\documentclass{article}" "\n"
                    rf"\title{{Paper {i}: {desc}}}" "\n"
                    r"\begin{document}" "\n"
                    rf"Analysis of {domain_hint} with formula:" "\n"
                    r"\begin{equation}" "\n"
                    rf"{formula}" "\n"
                    r"\end{equation}" "\n"
                    r"\end{document}"
                )
            p.write_text(content, encoding="utf-8")
            self.corpus_files.append(p)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_workers_1_4_8_deterministic_equivalence(self):
        """Verify that sequential (1) and concurrent (4, 8) workers produce identical records."""
        config_seq = ExtractorConfig(mock=True, max_workers=1)
        config_p4 = ExtractorConfig(mock=True, max_workers=4)
        config_p8 = ExtractorConfig(mock=True, max_workers=8)

        pipeline_seq = ExtractionPipeline(config=config_seq)
        pipeline_p4 = ExtractionPipeline(config=config_p4)
        pipeline_p8 = ExtractionPipeline(config=config_p8)

        report_seq = pipeline_seq.run_batch(self.corpus_files)
        report_p4 = pipeline_p4.run_batch(self.corpus_files)
        report_p8 = pipeline_p8.run_batch(self.corpus_files)

        # 1. Counts must match
        self.assertEqual(len(report_seq), len(self.corpus_files))
        self.assertEqual(len(report_p4), len(self.corpus_files))
        self.assertEqual(len(report_p8), len(self.corpus_files))

        # 2. File ordering must be strictly identical (alphabetical sort guarantee)
        files_seq = [r.file for r in report_seq]
        files_p4 = [r.file for r in report_p4]
        files_p8 = [r.file for r in report_p8]
        self.assertEqual(files_seq, files_p4)
        self.assertEqual(files_seq, files_p8)

        # 3. Domain and bound content must match 1:1 across all records
        for i in range(len(report_seq)):
            r_seq = report_seq[i]
            r_p4 = report_p4[i]
            r_p8 = report_p8[i]

            self.assertEqual(r_seq.file, r_p4.file)
            self.assertEqual(r_seq.domain, r_p4.domain)
            self.assertEqual(r_seq.domain, r_p8.domain)

            # Compare extracted formulas
            formulas_seq = [b.formula for b in r_seq.bounds]
            formulas_p4 = [b.formula for b in r_p4.bounds]
            formulas_p8 = [b.formula for b in r_p8.bounds]
            self.assertEqual(formulas_seq, formulas_p4)
            self.assertEqual(formulas_seq, formulas_p8)

            # Compare summaries
            self.assertEqual(r_seq.summary, r_p4.summary)
            self.assertEqual(r_seq.summary, r_p8.summary)

    def test_high_worker_pool_on_small_batch(self):
        """Stress-test thread-pool sizing when max_workers exceeds batch size."""
        small_batch = self.corpus_files[:3]

        # max_workers = 16 and 32
        p16 = ExtractionPipeline(config=ExtractorConfig(mock=True, max_workers=16))
        p32 = ExtractionPipeline(config=ExtractorConfig(mock=True, max_workers=32))

        rep16 = p16.run_batch(small_batch)
        rep32 = p32.run_batch(small_batch)

        self.assertEqual(len(rep16), 3)
        self.assertEqual(len(rep32), 3)
        self.assertEqual([r.file for r in rep16], [r.file for r in rep32])

    def test_duplicate_basename_in_subdirectories_determinism(self):
        """Examine behavior when two files in different subdirectories have identical names.

        e.g., sub_a/paper.md and sub_b/paper.md.
        Both have r.file == 'paper.md'.
        """
        sub_a = self.root / "sub_a"
        sub_b = self.root / "sub_b"
        sub_a.mkdir()
        sub_b.mkdir()

        f_a = sub_a / "duplicate_name.md"
        f_b = sub_b / "duplicate_name.md"

        f_a.write_text("# Paper A\n$$\\lambda \\le 1$$", encoding="utf-8")
        f_b.write_text("# Paper B\n$$\\mu \\ge 2$$", encoding="utf-8")

        p = ExtractionPipeline(config=ExtractorConfig(mock=True, max_workers=2))
        rep = p.run_batch([f_a, f_b])

        self.assertEqual(len(rep), 2)
        # Both records have duplicate_name.md as filename, but distinct path attributes
        self.assertEqual(rep[0].file, "duplicate_name.md")
        self.assertEqual(rep[1].file, "duplicate_name.md")
        paths = {rep[0].path, rep[1].path}
        self.assertIn(str(f_a.resolve()), paths)
        self.assertIn(str(f_b.resolve()), paths)


class TestFaultToleranceAndErrorContainment(unittest.TestCase):
    """Hostile fault tolerance and error containment testing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = ExtractorConfig(mock=True)
        self.pipeline = ExtractionPipeline(config=self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mixed_chaos_batch_does_not_crash(self):
        """Batch containing valid files, non-existent files, and simulated crashes must succeed."""
        # 1. Valid file
        valid_1 = self.root / "valid_1.md"
        valid_1.write_text("# Valid 1\n$$\\nabla f(x) = 0$$\n**Theorem**: Optimality.", encoding="utf-8")

        # 2. Non-existent file
        missing_file = self.root / "does_not_exist_at_all.md"

        # 3. Directory path disguised as manuscript file
        fake_dir_file = self.root / "fake_directory.md"
        fake_dir_file.mkdir()

        # 4. Valid file 2
        valid_2 = self.root / "valid_2.tex"
        valid_2.write_text(
            r"\documentclass{article}\title{Valid 2}\begin{document}\[ x \le y \]\end{document}",
            encoding="utf-8",
        )

        # 5. File that triggers a simulated catastrophic exception in extractor
        exploding_file = self.root / "exploding.md"
        exploding_file.write_text("# Exploding file content", encoding="utf-8")

        batch = [valid_1, missing_file, fake_dir_file, valid_2, exploding_file]

        orig_extract_file = self.pipeline.extractor.extract_file

        def mocked_extract_file(path):
            p = Path(path)
            if p.name == "exploding.md":
                raise RuntimeError("Catastrophic simulated exception during parsing!")
            return orig_extract_file(path)

        with patch.object(self.pipeline.extractor, "extract_file", side_effect=mocked_extract_file):
            report = self.pipeline.run_batch(batch)

        # Verification:
        # Valid files must have succeeded
        self.assertEqual(len(report), 2)
        succeeded_files = {r.file for r in report}
        self.assertEqual(succeeded_files, {"valid_1.md", "valid_2.tex"})

        # Metadata must accurately reflect error metrics
        meta = report.metadata or {}
        self.assertEqual(meta.get("total_files"), 5)
        self.assertEqual(meta.get("successful"), 2)
        self.assertEqual(meta.get("failed"), 3)

        errors = meta.get("errors", [])
        self.assertEqual(len(errors), 3)

        failed_file_strings = [e["file"] for e in errors]
        self.assertTrue(any("does_not_exist_at_all.md" in f for f in failed_file_strings))
        self.assertTrue(any("fake_directory.md" in f for f in failed_file_strings))
        self.assertTrue(any("exploding.md" in f for f in failed_file_strings))

    def test_empty_and_whitespace_manuscripts_graceful_handling(self):
        """0-byte and whitespace-only files must not crash extractor or pipeline."""
        empty_file = self.root / "zero_bytes.md"
        empty_file.write_text("", encoding="utf-8")

        whitespace_file = self.root / "whitespace_only.tex"
        whitespace_file.write_text("   \n\t\n   ", encoding="utf-8")

        valid_file = self.root / "valid.md"
        valid_file.write_text("# Valid\nFormula $a \\le b$.", encoding="utf-8")

        report = self.pipeline.run_batch([empty_file, whitespace_file, valid_file])

        # All 3 files produce ExtractedRecord instances (empty files return graceful empty records)
        self.assertEqual(len(report), 3)
        self.assertEqual(report.metadata.get("successful"), 3)
        self.assertEqual(report.metadata.get("failed"), 0)

        # Inspect empty records
        empty_rec = [r for r in report if r.file == "zero_bytes.md"][0]
        self.assertEqual(empty_rec.bounds, [])
        self.assertIn("empty", empty_rec.summary.lower())

        ws_rec = [r for r in report if r.file == "whitespace_only.tex"][0]
        self.assertEqual(ws_rec.bounds, [])

    def test_complete_batch_failure_resilience(self):
        """When 100% of files in a batch fail, pipeline must produce an empty valid report."""
        non_existent_1 = self.root / "missing_1.md"
        non_existent_2 = self.root / "missing_2.md"
        out_json = self.root / "all_failed_report.json"

        report = self.pipeline.run_batch([non_existent_1, non_existent_2], output_file=out_json)

        self.assertEqual(len(report), 0)
        self.assertEqual(report.metadata.get("failed"), 2)
        self.assertEqual(report.metadata.get("successful"), 0)

        # Output JSON must exist, be valid JSON array, and be empty
        self.assertTrue(out_json.exists())
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)


class TestExclusionIntegrity(unittest.TestCase):
    """Hostile blacklist and noise exclusion testing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pipeline = ExtractionPipeline(config=ExtractorConfig(mock=True))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_strict_pruning_of_workspace_noise_directories(self):
        """Scaffold nested directories under node_modules, .agents, .git, build, dist.

        Verify that ZERO noise files are scanned or processed.
        """
        # 1. Valid files
        valid_root = self.root / "valid_root.md"
        valid_root.write_text("# Root Valid\nFormula $x \\ge 1$.", encoding="utf-8")

        valid_nested_dir = self.root / "deep" / "research"
        valid_nested_dir.mkdir(parents=True, exist_ok=True)
        valid_nested = valid_nested_dir / "valid_nested.tex"
        valid_nested.write_text(r"\documentclass{article}\begin{document}$y \le 2$\end{document}", encoding="utf-8")

        # 2. Blacklisted noise paths
        noise_dirs = [
            self.root / "node_modules",
            self.root / "frontend" / "node_modules" / "sub_pkg",
            self.root / ".agents" / "teamwork" / "agent_1",
            self.root / ".git" / "hooks",
            self.root / "build" / "temp",
            self.root / "dist" / "bundle",
            self.root / "__pycache__",
            self.root / ".venv" / "lib",
        ]

        noise_files = []
        for nd in noise_dirs:
            nd.mkdir(parents=True, exist_ok=True)
            nf_md = nd / "noise_manuscript.md"
            nf_tex = nd / "noise_manuscript.tex"
            nf_md.write_text("# Noise MD\nFormula $\\text{noise} = 1$.", encoding="utf-8")
            nf_tex.write_text(r"\documentclass{article}\begin{document}$noise=1$\end{document}", encoding="utf-8")
            noise_files.extend([nf_md, nf_tex])

        # Run extraction from the root
        report = self.pipeline.run(target_dir=self.root)

        # Verification: ONLY the 2 valid files must exist in report
        self.assertEqual(len(report), 2)
        extracted_files = [r.file for r in report]
        self.assertIn("valid_root.md", extracted_files)
        self.assertIn("valid_nested.tex", extracted_files)

        # Assert zero noise files made it into the report
        for nf in noise_files:
            self.assertNotIn(nf.name, extracted_files)
            for r in report:
                self.assertNotEqual(r.path, str(nf.resolve()))

    def test_custom_excludes_override(self):
        """Verify dynamic custom excludes parameter suppresses targeted folders."""
        secret_dir = self.root / "secret_experiments"
        secret_dir.mkdir(parents=True, exist_ok=True)
        secret_file = secret_dir / "secret.md"
        secret_file.write_text("# Secret\nFormula $k = 42$.", encoding="utf-8")

        public_file = self.root / "public.md"
        public_file.write_text("# Public\nFormula $p = 1$.", encoding="utf-8")

        report = self.pipeline.run(target_dir=self.root, excludes=["secret_experiments"])

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0].file, "public.md")


class TestSchemaEnforcementAndSerialization(unittest.TestCase):
    """Schema invariant validation and serialization resilience."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pipeline = ExtractionPipeline(config=ExtractorConfig(mock=True))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_serialized_json_strictly_conforms_to_4_key_schema(self):
        """Serialized report JSON must be a list of records with keys: file, domain, bounds, summary."""
        test_file = self.root / "test_schema.md"
        test_file.write_text(
            "# Title\n$$\\lambda \\le 10$$\n**Theorem**: Convergence is guaranteed.",
            encoding="utf-8",
        )
        out_json = self.root / "schema_verified.json"

        report = self.pipeline.run(target_dir=self.root, output_file=out_json)
        self.assertTrue(out_json.exists())

        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)

        record_dict = payload[0]
        # Invariant: 4 mandatory keys
        for key in ("file", "domain", "bounds", "summary"):
            self.assertIn(key, record_dict)

        self.assertIsInstance(record_dict["file"], str)
        self.assertIsInstance(record_dict["domain"], str)
        self.assertIsInstance(record_dict["bounds"], list)
        self.assertIsInstance(record_dict["summary"], str)

        # Bounds structure check
        self.assertTrue(len(record_dict["bounds"]) > 0)
        for b in record_dict["bounds"]:
            self.assertIsInstance(b, dict)
            for b_key in ("name", "formula", "type", "context"):
                self.assertIn(b_key, b)
                self.assertIsInstance(b[b_key], str)

    def test_schema_validation_error_on_missing_or_none_fields(self):
        """ExtractedRecord.from_dict must raise SchemaValidationError on missing/null mandatory fields."""
        valid_payload = {
            "file": "test.md",
            "domain": "ML_Optimization",
            "bounds": [{"name": "B", "formula": "x < 1", "type": "bound", "context": ""}],
            "summary": "Valid summary.",
        }

        # Valid baseline
        rec = ExtractedRecord.from_dict(valid_payload)
        self.assertEqual(rec.file, "test.md")

        # Test each missing mandatory field
        for missing_field in ("file", "domain", "bounds", "summary"):
            bad_payload = dict(valid_payload)
            del bad_payload[missing_field]
            with self.assertRaises(SchemaValidationError):
                ExtractedRecord.from_dict(bad_payload)

        # Test each None mandatory field
        for none_field in ("file", "domain", "bounds", "summary"):
            bad_payload = dict(valid_payload)
            bad_payload[none_field] = None
            with self.assertRaises(SchemaValidationError):
                ExtractedRecord.from_dict(bad_payload)

    def test_concurrent_save_race_condition(self):
        """Multiple threads saving simultaneously to the exact same file path must not crash."""
        out_json = self.root / "concurrent_shared.json"

        report1 = ExtractionReport(records=[
            ExtractedRecord(file="f1.md", domain="ML_Optimization", bounds=[], summary="S1")
        ])
        report2 = ExtractionReport(records=[
            ExtractedRecord(file="f2.md", domain="Graph_Combinatorics", bounds=[], summary="S2")
        ])

        def save_report(rep):
            return rep.save(out_json)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(save_report, report1 if i % 2 == 0 else report2)
                for i in range(10)
            ]
            for fut in futures:
                fut.result()  # Must not raise PermissionError or collision error

        # File must exist and be valid JSON
        self.assertTrue(out_json.exists())
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
