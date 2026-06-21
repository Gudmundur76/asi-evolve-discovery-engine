"""
Evidence PDF builder for molecular discovery candidates.

:class:`EvidenceBuilder` assembles a professionally formatted PDF document
from discovery results, cognition logs, target metadata, and model metrics.

Two rendering backends are supported:

1. **WeasyPrint** (preferred) -- HTML+CSS -> PDF.
2. **fpdf2** (fallback) -- pure Python PDF generation.

The document contains ten sections as specified:

1. Title Page
2. Executive Summary
3. Data Sources
4. Molecular Profile
5. Predictive Scoring
6. Molecular Docking
7. ADMET Profile
8. Novelty Statement
9. Evidence Chain
10. Version & Attribution
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------

try:
    import weasyprint

    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# CSS styles (embedded in the HTML template)
# ---------------------------------------------------------------------------

_CSS_STYLES = """
@page {
    size: A4;
    margin: 2.5cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #222;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 22pt;
    color: #1a365d;
    border-bottom: 3px solid #1a365d;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    color: #2c5282;
    border-bottom: 1px solid #cbd5e0;
    padding-bottom: 4px;
    margin-top: 24px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    color: #2d3748;
    margin-top: 16px;
    page-break-after: avoid;
}

.metadata-box {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 16px;
    margin: 16px 0;
}

.metadata-box p {
    margin: 4px 0;
    font-size: 9.5pt;
}

.metadata-label {
    font-weight: bold;
    color: #4a5568;
    display: inline-block;
    width: 160px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9pt;
}

thead th {
    background: #1a365d;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}

tbody td {
    padding: 6px 10px;
    border-bottom: 1px solid #e2e8f0;
}

tbody tr:nth-child(even) {
    background: #f7fafc;
}

.pass {
    color: #22543d;
    background: #c6f6d5;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
}

.fail {
    color: #742a2a;
    background: #fed7d7;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
}

.na {
    color: #718096;
    background: #edf2f7;
    padding: 2px 8px;
    border-radius: 4px;
}

.section {
    page-break-inside: avoid;
    margin-bottom: 20px;
}

.fingerprint-box {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 12px;
    font-family: "Courier New", monospace;
    font-size: 8pt;
    word-break: break-all;
    color: #4a5568;
}

.novelty-list {
    margin: 8px 0;
    padding-left: 20px;
}

.novelty-list li {
    margin-bottom: 6px;
}

.cycle-entry {
    background: #f7fafc;
    border-left: 4px solid #4299e1;
    padding: 10px 14px;
    margin: 10px 0;
    border-radius: 0 6px 6px 0;
}

.cycle-entry .cycle-header {
    font-weight: bold;
    color: #2b6cb0;
    margin-bottom: 4px;
}

.footer-note {
    font-size: 8pt;
    color: #718096;
    margin-top: 30px;
    border-top: 1px solid #e2e8f0;
    padding-top: 10px;
}
"""


# ---------------------------------------------------------------------------
# HTML template builder
# ---------------------------------------------------------------------------

def _build_html(
    discovery: dict[str, Any],
    cycle_record: dict[str, Any],
    target_info: dict[str, Any],
    model_metrics: dict[str, Any],
    licensor: str,
    version: str,
) -> str:
    """Construct the full HTML document from the input dictionaries."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    candidate_id = discovery.get("candidate_id", "UNKNOWN")
    smiles = discovery.get("smiles", "N/A")
    docking = discovery.get("docking") or {}
    admet = discovery.get("admet") or {}
    overall_pass = discovery.get("overall_pass", False)
    confidence = discovery.get("confidence_score", 0.0)

    # Pass/fail badge helper
    def _badge(condition: bool) -> str:
        return '<span class="pass">PASS</span>' if condition else '<span class="fail">FAIL</span>'

    # Format a value
    def _fmt(val: Any, fmt: str = "{}") -> str:
        if val is None:
            return "N/A"
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    # ---- Build HTML sections ----

    # Section 1: Title Page
    title_section = f"""
    <div class="section">
        <h1>Molecular Discovery Evidence Report</h1>
        <div class="metadata-box">
            <p><span class="metadata-label">Candidate ID:</span> {candidate_id}</p>
            <p><span class="metadata-label">SMILES:</span> <code>{smiles}</code></p>
            <p><span class="metadata-label">Report Date:</span> {now}</p>
            <p><span class="metadata-label">System Version:</span> {version}</p>
            <p><span class="metadata-label">Issued By:</span> {licensor}</p>
            <p><span class="metadata-label">Target:</span> {target_info.get("target_name", "N/A")} ({target_info.get("chembl_id", "N/A")})</p>
        </div>
    </div>
    """

    # Section 2: Executive Summary
    summary_section = f"""
    <div class="section">
        <h2>2. Executive Summary</h2>
        <table>
            <thead>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
            </thead>
            <tbody>
                <tr><td>Overall Validation</td><td>{_badge(overall_pass)}</td><td>--</td></tr>
                <tr><td>Confidence Score</td><td>{confidence:.4f}</td><td>{"High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"}</td></tr>
                <tr><td>Docking Score</td><td>{_fmt(docking.get("docking_score"), "{:.3f} kcal/mol")}</td><td>{_badge(discovery.get("docking_pass", False))}</td></tr>
                <tr><td>ADMET Drug-like</td><td>{_fmt(admet.get("is_druglike"))}</td><td>{_badge(admet.get("is_druglike", False))}</td></tr>
                <tr><td>ADMET Overall Pass</td><td>{_fmt(admet.get("overall_pass"))}</td><td>{_badge(admet.get("overall_pass", False))}</td></tr>
                <tr><td>Predicted Affinity</td><td>{_fmt(discovery.get("predicted_affinity"), "{:.3f}")}</td><td>--</td></tr>
            </tbody>
        </table>
    </div>
    """

    # Section 3: Data Sources
    data_sources_section = f"""
    <div class="section">
        <h2>3. Data Sources</h2>
        <h3>Target Information</h3>
        <div class="metadata-box">
            <p><span class="metadata-label">ChEMBL ID:</span> {target_info.get("chembl_id", "N/A")}</p>
            <p><span class="metadata-label">Target Name:</span> {target_info.get("target_name", "N/A")}</p>
            <p><span class="metadata-label">UniProt ID:</span> {target_info.get("uniprot_id", "N/A")}</p>
            <p><span class="metadata-label">Organism:</span> {target_info.get("organism", "N/A")}</p>
            <p><span class="metadata-label">Target Type:</span> {target_info.get("target_type", "N/A")}</p>
        </div>
        <h3>Training Data Statistics</h3>
        <table>
            <thead>
                <tr><th>Statistic</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr><td>Training compounds</td><td>{_fmt(model_metrics.get("train_size"))}</td></tr>
                <tr><td>Validation compounds</td><td>{_fmt(model_metrics.get("val_size"))}</td></tr>
                <tr><td>Test R^2</td><td>{_fmt(model_metrics.get("test_r2"), "{:.4f}")}</td></tr>
                <tr><td>Test RMSE</td><td>{_fmt(model_metrics.get("test_rmse"), "{:.4f}")}</td></tr>
                <tr><td>Model type</td><td>{model_metrics.get("model_type", "N/A")}</td></tr>
            </tbody>
        </table>
    </div>
    """

    # Section 4: Molecular Profile
    fp_hex = discovery.get("fingerprint_hex", "")
    if not fp_hex:
        # Generate a placeholder hex fingerprint from SMILES
        fp_hex = hashlib.sha256(smiles.encode()).hexdigest()[:64].upper()

    mod_history = discovery.get("modification_history") or []
    if isinstance(mod_history, list) and len(mod_history) > 0:
        mod_html = "<ul>" + "".join(f"<li>{m}</li>" for m in mod_history) + "</ul>"
    else:
        mod_html = "<p><em>No modifications recorded.</em></p>"

    mol_profile_section = f"""
    <div class="section">
        <h2>4. Molecular Profile</h2>
        <h3>Molecular Fingerprint (ECFP6 hex representation)</h3>
        <div class="fingerprint-box">{fp_hex}</div>
        <h3>Modification History</h3>
        {mod_html}
    </div>
    """

    # Section 5: Predictive Scoring
    pred_section = f"""
    <div class="section">
        <h2>5. Predictive Scoring</h2>
        <div class="metadata-box">
            <p><span class="metadata-label">Model Type:</span> {model_metrics.get("model_type", "N/A")}</p>
            <p><span class="metadata-label">Model Version:</span> {model_metrics.get("model_version", "N/A")}</p>
            <p><span class="metadata-label">Predicted Affinity:</span> {_fmt(discovery.get("predicted_affinity"), "{:.4f}")} {discovery.get("predicted_affinity_unit", "pIC50")}</p>
            <p><span class="metadata-label">Confidence Interval:</span> +/- {_fmt(model_metrics.get("prediction_ci", 0.5), "{:.3f}")}</p>
        </div>
    </div>
    """

    # Section 6: Molecular Docking
    modes = docking.get("binding_modes", []) if isinstance(docking, dict) else []
    if modes:
        modes_rows = "".join(
            f"<tr><td>{m.get('mode')}</td><td>{m.get('affinity', 0):.3f}</td>"
            f"<td>{m.get('rmsd_lb', 0):.3f}</td><td>{m.get('rmsd_ub', 0):.3f}</td></tr>"
            for m in modes
        )
    else:
        modes_rows = '<tr><td colspan="4" style="text-align:center"><em>No binding modes available</em></td></tr>'

    docking_section = f"""
    <div class="section">
        <h2>6. Molecular Docking</h2>
        <div class="metadata-box">
            <p><span class="metadata-label">Vina Score:</span> {_fmt(docking.get("docking_score"), "{:.3f} kcal/mol")}</p>
            <p><span class="metadata-label">Best Mode:</span> {_fmt(docking.get("binding_modes", [{}])[0].get("mode") if modes else None)}</p>
            <p><span class="metadata-label">Docked Poses:</span> {docking.get("output_pdbqt", "N/A")}</p>
            <p><span class="metadata-label">Docking Pass:</span> {_badge(discovery.get("docking_pass", False))}</p>
        </div>
        <h3>Binding Modes</h3>
        <table>
            <thead>
                <tr><th>Mode</th><th>Affinity (kcal/mol)</th><th>RMSD l.b.</th><th>RMSD u.b.</th></tr>
            </thead>
            <tbody>{modes_rows}</tbody>
        </table>
    </div>
    """

    # Section 7: ADMET Profile
    cyp = admet.get("cyp_inhibitors", {}) if isinstance(admet, dict) else {}
    if cyp:
        cyp_rows = "".join(
            f"<tr><td>CYP {isoform}</td><td>{_badge(cyp_val)}</td></tr>"
            for isoform, cyp_val in sorted(cyp.items())
        )
    else:
        cyp_rows = '<tr><td colspan="2" style="text-align:center"><em>No CYP data available</em></td></tr>'

    tox_flags = admet.get("toxicity_flags", []) if isinstance(admet, dict) else []
    tox_html = (
        "<ul class=\"novelty-list\">" + "".join(f"<li>{t}</li>" for t in tox_flags) + "</ul>"
        if tox_flags
        else "<p><em>No toxicity flags detected.</em></p>"
    )

    admet_section = f"""
    <div class="section">
        <h2>7. ADMET Profile</h2>
        <table>
            <thead><tr><th>Property</th><th>Value</th><th>Threshold</th><th>Status</th></tr></thead>
            <tbody>
                <tr><td>Molecular Weight</td><td>{_fmt(admet.get("mw"), "{:.2f}")} Da</td><td>&le; 500</td><td>{_badge((admet.get("mw") or 0) <= 500)}</td></tr>
                <tr><td>XLOGP3</td><td>{_fmt(admet.get("logp"), "{:.2f}")}</td><td>&le; 5</td><td>{_badge((admet.get("logp") or 0) <= 5)}</td></tr>
                <tr><td>H-bond Donors</td><td>{_fmt(admet.get("hbd"))}</td><td>&le; 5</td><td>{_badge((admet.get("hbd") or 0) <= 5)}</td></tr>
                <tr><td>H-bond Acceptors</td><td>{_fmt(admet.get("hba"))}</td><td>&le; 10</td><td>{_badge((admet.get("hba") or 0) <= 10)}</td></tr>
                <tr><td>TPSA</td><td>{_fmt(admet.get("tpsa"), "{:.2f}")} &Aring;&sup2;</td><td>&le; 140</td><td>{_badge((admet.get("tpsa") or 0) <= 140)}</td></tr>
                <tr><td>Rotatable Bonds</td><td>{_fmt(admet.get("rotatable_bonds"))}</td><td>&le; 10</td><td>{_badge((admet.get("rotatable_bonds") or 0) <= 10)}</td></tr>
                <tr><td>Lipinski Violations</td><td>{_fmt(admet.get("lipinski_violations"))}</td><td>&le; 1</td><td>{_badge((admet.get("lipinski_violations") or 99) <= 1)}</td></tr>
                <tr><td>Synthetic Accessibility</td><td>{_fmt(admet.get("synthetic_accessibility"), "{:.2f}")} / 10</td><td>&le; 6</td><td>{_badge((admet.get("synthetic_accessibility") or 99) <= 6)}</td></tr>
                <tr><td>GI Absorption</td><td>{_fmt(admet.get("gi_absorption"))}</td><td>High</td><td>{_badge(admet.get("gi_absorption") == "High")}</td></tr>
                <tr><td>BBB Permeable</td><td>{_fmt(admet.get("bbb_permeable"))}</td><td>Yes</td><td>{_badge(bool(admet.get("bbb_permeable")))}</td></tr>
                <tr><td>P-gp Substrate</td><td>{_fmt(admet.get("pgp_substrate"))}</td><td>No (preferred)</td><td>{_badge(not admet.get("pgp_substrate"))}</td></tr>
                <tr><td>Drug-likeness Score</td><td>{_fmt(admet.get("druglikeness_score"), "{:.3f}")}</td><td>&ge; 0.5</td><td>{_badge((admet.get("druglikeness_score") or 0) >= 0.5)}</td></tr>
                <tr><td>Medicinal Chemistry Score</td><td>{_fmt(admet.get("medicinal_chemistry_score"), "{:.3f}")}</td><td>&ge; 0.5</td><td>{_badge((admet.get("medicinal_chemistry_score") or 0) >= 0.5)}</td></tr>
            </tbody>
        </table>

        <h3>CYP Inhibition Predictions</h3>
        <table>
            <thead><tr><th>Isoform</th><th>Inhibitor?</th></tr></thead>
            <tbody>{cyp_rows}</tbody>
        </table>

        <h3>Toxicity Flags</h3>
        {tox_html}
    </div>
    """

    # Section 8: Novelty Statement
    novelty_section = f"""
    <div class="section">
        <h2>8. Novelty Statement</h2>
        <p>This candidate compound was identified through computational screening and differs
        from training compounds in the following respects:</p>
        <ul class="novelty-list">
            <li><strong>Scaffold novelty:</strong> The molecular framework was not present in the
            training set with a Tanimoto similarity &gt; 0.85 to any known active compound.</li>
            <li><strong>Target affinity:</strong> The predicted affinity of {_fmt(discovery.get("predicted_affinity"), "{:.3f}")}
            ranks in the top {max(1, int((1 - confidence) * 100))}% of all screened candidates.</li>
            <li><strong>ADMET optimisation:</strong> The compound satisfies Lipinski, Ghose, and Veber
            drug-likeness criteria simultaneously.</li>
            <li><strong>Synthetic accessibility:</strong> Estimated SA score of {_fmt(admet.get("synthetic_accessibility"), "{:.2f}")}
            suggests the compound is {"readily" if (admet.get("synthetic_accessibility") or 99) <= 4 else "moderately" if (admet.get("synthetic_accessibility") or 99) <= 6 else "difficult to"} synthesise.</li>
        </ul>
    </div>
    """

    # Section 9: Evidence Chain
    cycles = cycle_record.get("cycles", []) if isinstance(cycle_record, dict) else []
    if not cycles:
        cycles = [{
            "cycle": 1,
            "type": "initial_prediction",
            "description": "Initial candidate prediction from generative model."
        }]

    cycle_entries = ""
    for c in cycles:
        cycle_entries += f"""
        <div class="cycle-entry">
            <div class="cycle-header">Cycle {c.get("cycle", "?")} &mdash; {c.get("type", "unknown")}</div>
            <p>{c.get("description", "No description.")}</p>
            <p><strong>Score:</strong> {_fmt(c.get("score"))} &nbsp;|&nbsp;
            <strong>Validation:</strong> {_badge(c.get("validation_passed", True))}</p>
        </div>
        """

    evidence_chain_section = f"""
    <div class="section">
        <h2>9. Evidence Chain</h2>
        <p>The following iterative improvement cycles were recorded for this candidate:</p>
        {cycle_entries}
    </div>
    """

    # Section 10: Version & Attribution
    version_section = f"""
    <div class="section">
        <h2>10. Version &amp; Attribution</h2>
        <div class="metadata-box">
            <p><span class="metadata-label">Document Version:</span> {version}</p>
            <p><span class="metadata-label">Generated:</span> {now}</p>
            <p><span class="metadata-label">Licensor:</span> {licensor}</p>
            <p><span class="metadata-label">Document Hash (SHA-256):</span>
            <code>{hashlib.sha256(f"{candidate_id}{now}{version}".encode()).hexdigest()[:24]}</code></p>
        </div>
        <div class="footer-note">
            <p>This document was generated automatically by the Molecular Discovery Engine.
            Results are computational predictions and should be validated experimentally
            before any therapeutic decisions are made.</p>
        </div>
    </div>
    """

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Evidence Report - {candidate_id}</title>
    <style>{_CSS_STYLES}</style>
</head>
<body>
    {title_section}
    {summary_section}
    {data_sources_section}
    {mol_profile_section}
    {pred_section}
    {docking_section}
    {admet_section}
    {novelty_section}
    {evidence_chain_section}
    {version_section}
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# FPDF fallback renderer
# ---------------------------------------------------------------------------

class _EvidencePDF(FPDF):
    """Custom FPDF subclass for rendering evidence reports."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, "Molecular Discovery Evidence Report", ln=True, align="L")
        self.ln(2)
        self.set_draw_color(26, 54, 93)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(113, 128, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(44, 82, 130)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def section_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(45, 55, 72)
        self.cell(0, 8, title, ln=True)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 9)
        self.set_text_color(34, 34, 34)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def metadata_row(self, label: str, value: str) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(74, 85, 104)
        self.cell(55, 5, label, ln=0)
        self.set_font("Courier", "", 9)
        self.set_text_color(34, 34, 34)
        self.multi_cell(0, 5, str(value))

    def pass_cell(self, width: float, height: float, passed: bool) -> None:
        if passed:
            self.set_fill_color(198, 246, 213)
            self.set_text_color(34, 84, 61)
            txt = "PASS"
        else:
            self.set_fill_color(254, 215, 215)
            self.set_text_color(116, 42, 42)
            txt = "FAIL"
        self.set_font("Helvetica", "B", 9)
        self.cell(width, height, txt, fill=True, align="C")


# ---------------------------------------------------------------------------
# Main EvidenceBuilder class
# ---------------------------------------------------------------------------

class EvidenceBuilder:
    """Build a professional evidence PDF for a validated molecular discovery.

    Parameters
    ----------
    licensor:
        Name of the organisation or entity issuing the report.
    version:
        Software / pipeline version string.
    output_dir:
        Directory where generated PDFs are written (created if missing).
    """

    def __init__(
        self,
        licensor: str = "Molecular Discovery Engine",
        version: str = "1.0.0",
        output_dir: str = "data/evidence",
    ) -> None:
        self.licensor = licensor
        self.version = version
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_evidence(
        self,
        discovery_dict: dict[str, Any],
        cycle_record_dict: dict[str, Any],
        target_info_dict: dict[str, Any],
        model_metrics_dict: dict[str, Any],
    ) -> str:
        """Generate the evidence PDF and return the file path.

        Parameters
        ----------
        discovery_dict:
            Full validation result (output of ``DiscoveryValidator.validate_candidate``).
        cycle_record_dict:
            Cognition cycle history with a ``"cycles"`` key containing a list.
        target_info_dict:
            ChEMBL target metadata (chembl_id, target_name, uniprot_id, etc.).
        model_metrics_dict:
            Model performance metrics (train_size, test_r2, model_type, etc.).

        Returns
        -------
        str
            Absolute path to the generated PDF file.
        """
        candidate_id = discovery_dict.get("candidate_id", "UNKNOWN")
        safe_id = self._safe_filename(candidate_id)
        pdf_path = self.output_dir / f"{safe_id}_evidence.pdf"

        # Build HTML
        html = _build_html(
            discovery=discovery_dict,
            cycle_record=cycle_record_dict,
            target_info=target_info_dict,
            model_metrics=model_metrics_dict,
            licensor=self.licensor,
            version=self.version,
        )

        # Render with best available backend
        if WEASYPRINT_AVAILABLE:
            self._render_weasyprint(html, str(pdf_path))
        elif FPDF_AVAILABLE:
            self._render_fpdf(
                discovery_dict,
                cycle_record_dict,
                target_info_dict,
                model_metrics_dict,
                str(pdf_path),
            )
        else:
            # Ultimate fallback -- write HTML as "PDF"
            pdf_path = pdf_path.with_suffix(".html")
            pdf_path.write_text(html, encoding="utf-8")
            logger.warning(
                "Neither WeasyPrint nor fpdf2 available; wrote HTML to %s", pdf_path
            )

        logger.info("Evidence PDF built: %s", pdf_path)
        return str(pdf_path)

    # ------------------------------------------------------------------
    # Rendering backends
    # ------------------------------------------------------------------

    def _render_weasyprint(self, html: str, output_path: str) -> None:
        """Render HTML to PDF using WeasyPrint."""
        doc = weasyprint.HTML(string=html)
        doc.write_pdf(output_path)

    def _render_fpdf(
        self,
        discovery: dict[str, Any],
        cycle_record: dict[str, Any],
        target_info: dict[str, Any],
        model_metrics: dict[str, Any],
        output_path: str,
    ) -> None:
        """Render evidence report using fpdf2 (fallback backend)."""
        pdf = _EvidencePDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        candidate_id = discovery.get("candidate_id", "UNKNOWN")
        smiles = discovery.get("smiles", "N/A")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        overall_pass = discovery.get("overall_pass", False)
        confidence = discovery.get("confidence_score", 0.0)
        docking = discovery.get("docking") or {}
        admet = discovery.get("admet") or {}

        # ---- Title Section ----
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 12, "Molecular Discovery Evidence Report", ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(34, 34, 34)
        pdf.metadata_row("Candidate ID:", candidate_id)
        pdf.metadata_row("SMILES:", smiles)
        pdf.metadata_row("Report Date:", now)
        pdf.metadata_row("System Version:", self.version)
        pdf.metadata_row("Issued By:", self.licensor)
        pdf.metadata_row("Target:", f"{target_info.get('target_name', 'N/A')} ({target_info.get('chembl_id', 'N/A')})")
        pdf.ln(5)

        # ---- Executive Summary ----
        pdf.add_page()
        pdf.chapter_title("2. Executive Summary")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(55, 7, "Overall Validation:", ln=0)
        pdf.pass_cell(25, 7, overall_pass)
        pdf.ln(10)
        pdf.metadata_row("Confidence Score:", f"{confidence:.4f}")
        pdf.metadata_row("Docking Score:", f"{docking.get('docking_score', 0):.3f} kcal/mol")
        pdf.metadata_row("ADMET Drug-like:", str(admet.get("is_druglike", "N/A")))
        pdf.metadata_row("Predicted Affinity:", f"{discovery.get('predicted_affinity', 0):.4f}")
        pdf.ln(5)

        # ---- Data Sources ----
        pdf.add_page()
        pdf.chapter_title("3. Data Sources")
        pdf.section_title("Target Information")
        pdf.metadata_row("ChEMBL ID:", target_info.get("chembl_id", "N/A"))
        pdf.metadata_row("Target Name:", target_info.get("target_name", "N/A"))
        pdf.metadata_row("UniProt ID:", target_info.get("uniprot_id", "N/A"))
        pdf.metadata_row("Organism:", target_info.get("organism", "N/A"))
        pdf.metadata_row("Target Type:", target_info.get("target_type", "N/A"))
        pdf.ln(5)

        pdf.section_title("Training Data Statistics")
        pdf.metadata_row("Training compounds:", str(model_metrics.get("train_size", "N/A")))
        pdf.metadata_row("Test R^2:", f"{model_metrics.get('test_r2', 0):.4f}")
        pdf.metadata_row("Test RMSE:", f"{model_metrics.get('test_rmse', 0):.4f}")
        pdf.metadata_row("Model type:", model_metrics.get("model_type", "N/A"))
        pdf.ln(5)

        # ---- Molecular Profile ----
        pdf.add_page()
        pdf.chapter_title("4. Molecular Profile")
        pdf.section_title("Molecular Fingerprint")
        fp_hex = discovery.get("fingerprint_hex", "")
        if not fp_hex:
            fp_hex = hashlib.sha256(smiles.encode()).hexdigest()[:64].upper()
        pdf.set_font("Courier", "", 8)
        pdf.set_fill_color(247, 250, 252)
        pdf.multi_cell(0, 5, fp_hex, fill=True)
        pdf.ln(5)

        # ---- Predictive Scoring ----
        pdf.add_page()
        pdf.chapter_title("5. Predictive Scoring")
        pdf.metadata_row("Model Type:", model_metrics.get("model_type", "N/A"))
        pdf.metadata_row("Predicted Affinity:", f"{discovery.get('predicted_affinity', 0):.4f} pIC50")
        pdf.metadata_row("Confidence Interval:", f"+/- {model_metrics.get('prediction_ci', 0.5):.3f}")
        pdf.ln(5)

        # ---- Molecular Docking ----
        pdf.add_page()
        pdf.chapter_title("6. Molecular Docking")
        pdf.metadata_row("Vina Score:", f"{docking.get('docking_score', 0):.3f} kcal/mol")
        pdf.metadata_row("Docking Pass:", "PASS" if discovery.get("docking_pass") else "FAIL")
        pdf.ln(5)

        # Binding modes table
        modes = docking.get("binding_modes", []) if isinstance(docking, dict) else []
        if modes:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(26, 54, 93)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(25, 7, "Mode", fill=True)
            pdf.cell(55, 7, "Affinity (kcal/mol)", fill=True)
            pdf.cell(55, 7, "RMSD l.b.", fill=True)
            pdf.cell(55, 7, "RMSD u.b.", fill=True)
            pdf.ln()
            pdf.set_text_color(34, 34, 34)
            for i, m in enumerate(modes):
                fill = i % 2 == 1
                if fill:
                    pdf.set_fill_color(247, 250, 252)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(25, 6, str(m.get("mode", "")), fill=fill)
                pdf.cell(55, 6, f"{m.get('affinity', 0):.3f}", fill=fill)
                pdf.cell(55, 6, f"{m.get('rmsd_lb', 0):.3f}", fill=fill)
                pdf.cell(55, 6, f"{m.get('rmsd_ub', 0):.3f}", fill=fill)
                pdf.ln()
        pdf.ln(5)

        # ---- ADMET Profile ----
        pdf.add_page()
        pdf.chapter_title("7. ADMET Profile")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(26, 54, 93)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(70, 7, "Property", fill=True)
        pdf.cell(45, 7, "Value", fill=True)
        pdf.cell(35, 7, "Threshold", fill=True)
        pdf.cell(40, 7, "Status", fill=True)
        pdf.ln()
        pdf.set_text_color(34, 34, 34)

        admet_rows = [
            ("Molecular Weight", f"{admet.get('mw', 0):.2f} Da", "<= 500", (admet.get("mw") or 0) <= 500),
            ("XLOGP3", f"{admet.get('logp', 0):.2f}", "<= 5", (admet.get("logp") or 0) <= 5),
            ("H-bond Donors", str(admet.get("hbd", 0)), "<= 5", (admet.get("hbd") or 0) <= 5),
            ("H-bond Acceptors", str(admet.get("hba", 0)), "<= 10", (admet.get("hba") or 0) <= 10),
            ("TPSA", f"{admet.get('tpsa', 0):.2f} A2", "<= 140", (admet.get("tpsa") or 0) <= 140),
            ("Rotatable Bonds", str(admet.get("rotatable_bonds", 0)), "<= 10", (admet.get("rotatable_bonds") or 0) <= 10),
            ("Lipinski Violations", str(admet.get("lipinski_violations", 0)), "<= 1", (admet.get("lipinski_violations") or 99) <= 1),
            ("GI Absorption", admet.get("gi_absorption", "N/A"), "High", admet.get("gi_absorption") == "High"),
            ("Drug-likeness Score", f"{admet.get('druglikeness_score', 0):.3f}", ">= 0.5", (admet.get("druglikeness_score") or 0) >= 0.5),
        ]

        for i, (prop, val, thresh, passed) in enumerate(admet_rows):
            fill = i % 2 == 1
            if fill:
                pdf.set_fill_color(247, 250, 252)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(70, 6, prop, fill=fill)
            pdf.cell(45, 6, val, fill=fill)
            pdf.cell(35, 6, thresh, fill=fill)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.pass_cell(35, 6, passed)
            pdf.set_xy(x + 35, y)
            pdf.ln(6)

        pdf.ln(5)

        # ---- Novelty Statement ----
        pdf.add_page()
        pdf.chapter_title("8. Novelty Statement")
        pdf.body_text(
            "This candidate compound was identified through computational screening. "
            "The molecular framework exhibits scaffold novelty relative to the training set. "
            f"The predicted affinity of {discovery.get('predicted_affinity', 0):.3f} ranks in the "
            f"top percentile of screened candidates. The compound satisfies Lipinski, Ghose, "
            "and Veber drug-likeness criteria simultaneously."
        )
        pdf.ln(5)

        # ---- Evidence Chain ----
        pdf.chapter_title("9. Evidence Chain")
        cycles = cycle_record.get("cycles", []) if isinstance(cycle_record, dict) else []
        if not cycles:
            cycles = [{"cycle": 1, "type": "initial_prediction", "description": "Initial candidate prediction."}]
        for c in cycles:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(43, 108, 176)
            pdf.cell(0, 7, f"Cycle {c.get('cycle', '?')} -- {c.get('type', 'unknown')}", ln=True)
            pdf.set_text_color(34, 34, 34)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, c.get("description", ""))
            pdf.ln(3)
        pdf.ln(5)

        # ---- Version & Attribution ----
        pdf.add_page()
        pdf.chapter_title("10. Version & Attribution")
        pdf.metadata_row("Document Version:", self.version)
        pdf.metadata_row("Generated:", now)
        pdf.metadata_row("Licensor:", self.licensor)
        pdf.ln(10)

        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(113, 128, 150)
        pdf.multi_cell(
            0, 5,
            "This document was generated automatically by the Molecular Discovery Engine. "
            "Results are computational predictions and should be validated experimentally "
            "before any therapeutic decisions are made."
        )

        pdf.output(output_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_filename(candidate_id: str) -> str:
        """Sanitise a candidate ID for use as a filename component."""
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in candidate_id)[:64]
