import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf():
    pdf_path = r"results\CRAG_PROJECT_FORMULAS_CALCULATIONS_REPORT.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4
    )

    math_box_style = ParagraphStyle(
        'MathBox',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#742A2A'),
        backColor=colors.HexColor('#FFF5F5'),
        borderColor=colors.HexColor('#FEB2B2'),
        borderWidth=1,
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=5
    )

    story = []

    # Document Header
    story.append(Paragraph("Custom Corrective Retrieval-Augmented Generation (CRAG-CAP)", title_style))
    story.append(Paragraph("Master Technical Report: Formulas, Mathematical Calculations, Data Tables & Worked Examples", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1A365D'), spaceAfter=8))

    # Phase 1
    story.append(Paragraph("Phase 1: Problem Statement & CRAG 3-Way Core Concept", h1_style))
    story.append(Paragraph("Standard RAG blindly trusts retrieved text. When retrieval brings noisy or incorrect context, LLMs hallucinate. CRAG introduces a 3-Way Decision Controller evaluating passage relevance before generation:", body_style))

    p1_data = [
        [Paragraph("<b>Decision Pathway</b>", body_style), Paragraph("<b>Condition / Rule</b>", body_style), Paragraph("<b>Operational Execution Logic</b>", body_style)],
        [Paragraph("<b>CORRECT</b>", body_style), Paragraph("MaxScore >= +0.5920", body_style), Paragraph("High confidence. Refine local text; <b>SKIP web search</b>.", body_style)],
        [Paragraph("<b>AMBIGUOUS</b>", body_style), Paragraph("-0.9950 <= MaxScore < +0.5920", body_style), Paragraph("Uncertain confidence. Combine refined local text + <b>Serper Google Search</b>.", body_style)],
        [Paragraph("<b>INCORRECT</b>", body_style), Paragraph("MaxScore < -0.9950", body_style), Paragraph("Pure noise. Discard local text; execute query rewriting & <b>Serper Google Search</b>.", body_style)]
    ]
    t1 = Table(p1_data, colWidths=[90, 150, 300])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    # Phase 2
    story.append(Paragraph("Phase 2: Evaluator Model Training & Classification Performance", h1_style))
    story.append(Paragraph("We fine-tuned <b>google-t5/t5-small (60.5M parameters)</b> on PopQA training triples. Evaluator classification metrics:", body_style))

    p2_data = [
        [Paragraph("<b>Metric Name</b>", body_style), Paragraph("<b>Mathematical Formula</b>", body_style), Paragraph("<b>Evaluator Tested Result</b>", body_style)],
        [Paragraph("<b>Classification Accuracy</b>", body_style), Paragraph("Acc = (TP + TN) / (TP + TN + FP + FN)", body_style), Paragraph("<b>94.40%</b>", body_style)],
        [Paragraph("<b>Relevance Precision</b>", body_style), Paragraph("Precision = TP / (TP + FP)", body_style), Paragraph("<b>92.80%</b>", body_style)],
        [Paragraph("<b>Relevance Recall</b>", body_style), Paragraph("Recall = TP / (TP + FN)", body_style), Paragraph("<b>95.60%</b>", body_style)],
        [Paragraph("<b>Relevance F1-Score</b>", body_style), Paragraph("F1 = 2 * (Precision * Recall) / (Precision + Recall)", body_style), Paragraph("<b>94.18%</b>", body_style)]
    ]
    t2 = Table(p2_data, colWidths=[140, 250, 150])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    # Phase 3
    story.append(Paragraph("Phase 3: Mathematical Score Calculation & Score Adapter Derivation", h1_style))
    story.append(Paragraph("At decoder step 0, T5 outputs raw logits L1 (token '1', ID 209) and L0 (token '0', ID 3). Softmax normalizes them into relevance probability P(1):", body_style))
    story.append(Paragraph("Probability Normalization: P(1) = exp(L1) / [exp(L1) + exp(L0)]", math_box_style))
    story.append(Paragraph("Linear Score Adapter Derivation: S(P) = a * P(1) + b. Solving boundary conditions S(0) = -1.0 and S(1) = +1.0 yields:", body_style))
    story.append(Paragraph("CRAG_Score = 2.0 * P(1) - 1.0   in range [-1.0, +1.0]", math_box_style))

    story.append(Paragraph("<b>Worked Numerical Calculation Example:</b><br/>"
                           "• Raw T5 Logits: L1 = +4.50, L0 = +1.20<br/>"
                           "• Exponentials: exp(4.50) = 90.017, exp(1.20) = 3.320<br/>"
                           "• Relevance Probability: P(1) = 90.017 / (90.017 + 3.320) = 90.017 / 93.337 = <b>0.9644 (96.44%)</b><br/>"
                           "• CRAG Score Adapter: Score = 2.0 * 0.9644 - 1.0 = <b>+0.9288</b> (Triggers <b>CORRECT</b> path since +0.9288 >= +0.5920).", body_style))
    story.append(Spacer(1, 8))

    # Phase 4
    story.append(Paragraph("Phase 4: 3-Way Threshold Probability Reverse Mapping Math", h1_style))
    story.append(Paragraph("Reverse-mapping scores back to probabilities via P = (S + 1) / 2:", body_style))
    story.append(Paragraph("• Upper Threshold (gamma1 = +0.5920) ===> P(1) = (+0.5920 + 1) / 2 = 79.60% Probability (High Confidence)<br/>"
                           "• Lower Cutoff (gamma2 = -0.9950) ===> P(1) = (-0.9950 + 1) / 2 = 0.25% Probability (Pure Noise Cutoff)", math_box_style))

    # Phase 5 & 6
    story.append(Paragraph("Phase 5 & 6: Knowledge Processing & Infrastructure Integration", h1_style))
    story.append(Paragraph("• <b>Passage Decomposition</b>: Paragraphs split into sentence sub-strips (selection mode, ~1-3 sentences).<br/>"
                           "• <b>Sub-Strip Sentence Filtering</b>: Evaluator deletes any sub-strip with Score < -0.9950.<br/>"
                           "• <b>Search Query Rewriting</b>: Groq 120B LLM rewrites question: <i>\"What is Henry Feilden's occupation?\"</i> ===> <i>\"Henry Feilden occupation\"</i>.<br/>"
                           "• <b>Web Search API</b>: Real Google Serper API (Serper.dev) fetches live organic Google search snippets.<br/>"
                           "• <b>Multi-Key Auto-Rotation</b>: 4 Groq API keys (800,000 TPD) with HTTP 429 automatic retry backoff.", body_style))
    story.append(Spacer(1, 8))

    # Phase 7
    story.append(Paragraph("Phase 7: Full 1,385-Query Benchmark Results & Metric Formulas", h1_style))

    p7_data = [
        [Paragraph("<b>Metric Name</b>", body_style), Paragraph("<b>Mathematical Formula</b>", body_style), Paragraph("<b>Tested Result (N = 1,385)</b>", body_style)],
        [Paragraph("<b>Exact Match Accuracy (EM)</b>", body_style), Paragraph("EM = (Count of Exact Substring Matches) / N * 100", body_style), Paragraph("<b>58.63% (812 / 1,385)</b>", body_style)],
        [Paragraph("<b>QA Token Recall</b>", body_style), Paragraph("Recall = |Tokens(Pred) ∩ Tokens(Gold)| / |Tokens(Gold)|", body_style), Paragraph("<b>59.25%</b>", body_style)],
        [Paragraph("<b>QA Token Precision</b>", body_style), Paragraph("Precision = |Tokens(Pred) ∩ Tokens(Gold)| / |Tokens(Pred)|", body_style), Paragraph("<b>11.65%</b>", body_style)],
        [Paragraph("<b>QA Token F1-Score</b>", body_style), Paragraph("F1 = 2 * (Precision * Recall) / (Precision + Recall)", body_style), Paragraph("<b>18.87%</b>", body_style)],
        [Paragraph("<b>ROUGE-L F1-Score</b>", body_style), Paragraph("LCS-based F1 Similarity", body_style), Paragraph("<b>18.87%</b>", body_style)]
    ]
    t7 = Table(p7_data, colWidths=[140, 250, 150])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t7)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Pathway Metrics Breakdown (N = 1,385 Queries):</b>", body_style))

    p_path_data = [
        [Paragraph("<b>Pathway</b>", body_style), Paragraph("<b>Queries</b>", body_style), Paragraph("<b>% Dataset</b>", body_style), Paragraph("<b>EM Accuracy</b>", body_style), Paragraph("<b>Token Recall</b>", body_style), Paragraph("<b>Token F1</b>", body_style)],
        [Paragraph("<b>CORRECT</b>", body_style), Paragraph("583", body_style), Paragraph("42.09%", body_style), Paragraph("<b>70.50%</b>", body_style), Paragraph("<b>71.90%</b>", body_style), Paragraph("<b>22.66%</b>", body_style)],
        [Paragraph("<b>AMBIGUOUS</b>", body_style), Paragraph("661", body_style), Paragraph("47.73%", body_style), Paragraph("<b>48.41%</b>", body_style), Paragraph("<b>48.55%</b>", body_style), Paragraph("<b>15.38%</b>", body_style)],
        [Paragraph("<b>INCORRECT</b>", body_style), Paragraph("141", body_style), Paragraph("10.18%", body_style), Paragraph("<b>57.45%</b>", body_style), Paragraph("<b>57.09%</b>", body_style), Paragraph("<b>19.59%</b>", body_style)],
        [Paragraph("<b>TOTAL SYSTEM</b>", body_style), Paragraph("<b>1,385</b>", body_style), Paragraph("<b>100.00%</b>", body_style), Paragraph("<b>58.63%</b>", body_style), Paragraph("<b>59.25%</b>", body_style), Paragraph("<b>18.87%</b>", body_style)]
    ]
    t_path = Table(p_path_data, colWidths=[90, 60, 80, 100, 100, 110])
    t_path.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_path)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Benchmark Performance Comparison Table:</b>", body_style))
    p_comp_data = [
        [Paragraph("<b>System Architecture</b>", body_style), Paragraph("<b>Evaluator Model</b>", body_style), Paragraph("<b>Generator LLM</b>", body_style), Paragraph("<b>PopQA EM Accuracy</b>", body_style)],
        [Paragraph("Standard RAG Baseline", body_style), Paragraph("None", body_style), Paragraph("LLaMA-2 7B", body_style), Paragraph("44.80%", body_style)],
        [Paragraph("Original Published CRAG (Yan 2024)", body_style), Paragraph("T5-Large (770M)", body_style), Paragraph("Self-RAG 7B", body_style), Paragraph("53.90%", body_style)],
        [Paragraph("<b>OUR CUSTOM CRAG IMPLEMENTATION</b>", body_style), Paragraph("<b>T5-Small (60.5M)</b>", body_style), Paragraph("<b>Groq 120B</b>", body_style), Paragraph("<b>58.63% (+4.73% over paper)</b>", body_style)]
    ]
    t_comp = Table(p_comp_data, colWidths=[150, 120, 110, 160])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_comp)

    # Build Document
    doc.build(story)
    print(f"PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
