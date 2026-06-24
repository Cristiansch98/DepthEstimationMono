"""Render the SelfCalibDepth digest as an IEEE two-column A4 PDF, following the
structure of the ITEC-AP-2026 (IEEE conference) Word template:
spanning title + author block, Abstract, Keywords, Roman-numeral headings,
Table I, Fig. 1, and a References list.

    python src/make_pdf.py        # -> PAPER.pdf
"""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, FrameBreak, Image, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, Frame)
from reportlab.lib import colors

PAGE_W, PAGE_H = A4
LM = RM = 0.55 * inch
TM = 0.6 * inch
BM = 0.6 * inch
GAP = 0.25 * inch
COL_W = (PAGE_W - LM - RM - GAP) / 2
TITLE_H = 1.85 * inch
TOP = PAGE_H - TM

SERIF, SERIF_B, SERIF_I = "Times-Roman", "Times-Bold", "Times-Italic"

st_title = ParagraphStyle("title", fontName=SERIF_B, fontSize=19, leading=22, alignment=TA_CENTER)
st_auth = ParagraphStyle("auth", fontName=SERIF, fontSize=11, leading=13, alignment=TA_CENTER, spaceBefore=8)
st_aff = ParagraphStyle("aff", fontName=SERIF_I, fontSize=9, leading=11, alignment=TA_CENTER)
st_abs = ParagraphStyle("abs", fontName=SERIF_B, fontSize=9, leading=11, alignment=TA_JUSTIFY)
st_kw = ParagraphStyle("kw", fontName=SERIF_B, fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceBefore=4)
st_h1 = ParagraphStyle("h1", fontName=SERIF, fontSize=10, leading=12, alignment=TA_CENTER,
                       spaceBefore=10, spaceAfter=3)
st_h2 = ParagraphStyle("h2", fontName=SERIF_I, fontSize=10, leading=12, alignment=TA_LEFT,
                       spaceBefore=6, spaceAfter=2)
st_body = ParagraphStyle("body", fontName=SERIF, fontSize=10, leading=12, alignment=TA_JUSTIFY,
                         firstLineIndent=0.2 * inch)
st_cap = ParagraphStyle("cap", fontName=SERIF, fontSize=8, leading=9.5, alignment=TA_CENTER, spaceBefore=4)
st_tcap = ParagraphStyle("tcap", fontName=SERIF, fontSize=8, leading=9.5, alignment=TA_CENTER, spaceAfter=2)
st_ref = ParagraphStyle("ref", fontName=SERIF, fontSize=8, leading=9.5, alignment=TA_JUSTIFY,
                        leftIndent=12, firstLineIndent=-12)


def H1(n, t):
    return Paragraph(f"{n}.&nbsp;&nbsp;{t}", st_h1)


def H2(letter, t):
    return Paragraph(f"<i>{letter}.&nbsp;&nbsp;{t}</i>", st_h2)


def P(t):
    return Paragraph(t, st_body)


def build():
    doc = BaseDocTemplate("PAPER.pdf", pagesize=A4, leftMargin=LM, rightMargin=RM,
                          topMargin=TM, bottomMargin=BM, title="SelfCalibDepth Digest",
                          author="Cristian Cubides")
    title_frame = Frame(LM, TOP - TITLE_H, PAGE_W - LM - RM, TITLE_H, id="title",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    c1 = Frame(LM, BM, COL_W, TOP - TITLE_H - BM, id="c1", leftPadding=0, rightPadding=0)
    c2 = Frame(LM + COL_W + GAP, BM, COL_W, TOP - TITLE_H - BM, id="c2", leftPadding=0, rightPadding=0)
    f1 = Frame(LM, BM, COL_W, TOP - BM, id="f1", leftPadding=0, rightPadding=0)
    f2 = Frame(LM + COL_W + GAP, BM, COL_W, TOP - BM, id="f2", leftPadding=0, rightPadding=0)
    fw = Frame(LM, BM, PAGE_W - LM - RM, TOP - BM, id="fw", leftPadding=0, rightPadding=0)
    doc.addPageTemplates([PageTemplate(id="first", frames=[title_frame, c1, c2]),
                          PageTemplate(id="rest", frames=[f1, f2]),
                          PageTemplate(id="wide", frames=[fw])])

    s = []  # story
    # ---- spanning title block ----
    s.append(Paragraph("SelfCalibDepth: LiDAR-Supervised Self-Calibration for "
                       "Camera-Aware Monocular Metric Depth on Argoverse 2", st_title))
    s.append(Paragraph("Cristian Cubides", st_auth))
    s.append(Paragraph("Software Research and Development<br/>"
                       "SelfCalibDepth Project, Argoverse 2<br/>"
                       "cschrcm@gmail.com", st_aff))
    s.append(NextPageTemplate("rest"))
    s.append(FrameBreak())

    # ---- abstract + keywords (column 1) ----
    s.append(Paragraph(
        "<i>Abstract</i>&mdash;<i>We present SelfCalibDepth, a framework that learns metric "
        "distance from a single monocular image while simultaneously recovering the camera "
        "intrinsic parameters from that same image. A synchronized LiDAR sweep, projected "
        "into the camera, provides metric ground truth that anchors both a depth network and a "
        "learnable camera model. The two are coupled through a per-pixel ray map: the calibration "
        "defines the viewing rays, the rays condition a fine-tuned Depth Anything V2 backbone, "
        "the network predicts depth along each ray, and back-projecting with the learned "
        "intrinsics and comparing to LiDAR returns a gradient into the calibration. On the "
        "Argoverse 2 Sensor dataset, the best model recovers focal length to within 0.26% (fx) "
        "and 0.41% (fy) of the manufacturer calibration on held-out cameras, attains AbsRel 0.112, "
        "and estimates distance to vehicles within sixty meters to a mean absolute error of 8.6 m "
        "(3.9 m within thirty meters), all from a single image. A controlled ablation includes a "
        "negative result and fully annotated assumptions. Unifying four driving benchmarks "
        "(Argoverse 2, KITTI, nuScenes, Lyft L5) behind one interface, we further show that a "
        "twenty-frame few-shot adaptation transfers the model to KITTI (AbsRel 0.100, accuracy "
        "0.953) and nuScenes (0.122, 0.853), recovering focal length to about one percent.</i>",
        st_abs))
    s.append(Paragraph("<i>Keywords</i>&mdash;<i>monocular depth estimation; camera "
                       "self-calibration; LiDAR supervision; metric depth; Argoverse 2</i>", st_kw))

    # ---- I. Introduction ----
    s.append(H1("I", "Introduction"))
    s.append(P("Estimating absolute, metric distance from a single camera is ill-posed: a "
               "pinhole camera projects the three-dimensional world onto two dimensions and the "
               "overall scale is unobservable without a metric reference or a known camera. Two "
               "coupled unknowns make this hard: the depth of each pixel, and the camera "
               "intrinsics that relate pixels to viewing rays. Focal length in particular is "
               "entangled with metric scale, so a depth model that ignores intrinsics cannot "
               "generalize across cameras."))
    s.append(P("This work asks a single question: can a model, supervised only by LiDAR, learn "
               "to recreate the camera parameters from one image and use them to report the real "
               "distance to objects, particularly vehicles, and do so for cameras it was not "
               "calibrated on? We answer affirmatively. Contributions are: (1) a LiDAR-as-ground-"
               "truth pipeline turning motion-compensated sweeps into sparse metric depth and "
               "per-vehicle distances on Argoverse 2; (2) a self-calibrating, camera-aware depth "
               "model coupling a learnable camera model to a Depth Anything V2 backbone through a "
               "ray map; (3) a controlled study with an honest negative result and annotated "
               "assumptions."))

    # ---- II. Related Work ----
    s.append(H1("II", "Related Work"))
    s.append(P("<i>Monocular depth.</i> Supervised monocular depth dates to Eigen et al. [1] and "
               "the scale-invariant loss; MiDaS [2] and DPT [3] showed that relative "
               "(affine-invariant) depth transfers across datasets. Depth Anything V2 [4] scales "
               "this with a DINOv2 backbone and a DPT head. Relative depth is not metric: an "
               "affine ambiguity in inverse-depth remains."))
    s.append(P("<i>Metric and camera-aware depth.</i> Recovering metric scale needs known "
               "intrinsics or a learned prior. CAM-Convs [5] injects per-pixel calibration into "
               "convolutions; Metric3D [6] and ZoeDepth [7] recover metric scale by conditioning "
               "on the intrinsics. Our ray-map conditioning is in this lineage, but the "
               "intrinsics it consumes are learned, not given."))
    s.append(P("<i>Self-calibration.</i> DroidCalib [8] folds intrinsics into a deep "
               "bundle-adjustment layer, optimizing them jointly with structure from monocular "
               "video. We keep the jointly-optimized-intrinsics idea but replace the multi-view "
               "anchor with direct LiDAR, removing scale ambiguity. Argoverse 2 [9] uniquely "
               "provides per-camera ground-truth intrinsics and 3-D cuboids, letting us measure "
               "both depth and calibration error."))

    # ---- III. Methodology ----
    s.append(H1("III", "Methodology"))
    s.append(H2("A", "The ray-map coupling"))
    s.append(P("For each image-sweep pair we observe LiDAR points in the camera frame, which are "
               "independent of the intrinsics. Let theta be the camera model. For every pixel we "
               "compute a unit viewing ray that depends on theta. The network predicts depth "
               "along the ray; back-projecting gives a 3-D point. A 3-D point loss compares this "
               "to the LiDAR point: a wrong focal length yields points that miss LiDAR and a "
               "corrective gradient flows into theta. A reprojection loss requires LiDAR points "
               "projected with theta to land at the pixel whose depth predicts them. With a "
               "scale-invariant depth term these make calibration observable: a ten-percent focal "
               "error induces about a 0.46 m mean 3-D shift. Fig. 1 illustrates the coupling."))
    s.append(_fig("viz_v3/arch_diagram.png",
                  "Fig. 1.&nbsp;&nbsp;The ray-map coupling. The learnable intrinsics define the "
                  "viewing rays that condition the depth head; back-projecting the predicted depth "
                  "with the same intrinsics and matching LiDAR returns a gradient that corrects "
                  "the calibration."))
    s.append(H2("B", "Training objectives"))
    s.append(P("Training combines five terms. A scale-invariant log term (SILog) supervises the "
               "predicted depth at the LiDAR pixels. A 3-D point term back-projects the predicted "
               "depth with theta and matches the LiDAR points in the camera frame; because it "
               "depends on theta, it is the term that calibrates the camera. A reprojection term "
               "requires LiDAR points projected with theta to land at the pixel whose depth "
               "predicts them, further constraining the intrinsics. An edge-aware smoothness term "
               "regularizes the depth map, and weak priors keep the principal point near the image "
               "center and the distortion small. The depth terms shape the depth map; the 3-D and "
               "reprojection terms shape both the depth and theta, closing the self-calibration "
               "loop."))
    s.append(H2("C", "Camera-aware metric depth"))
    s.append(P("Depth Anything V2 outputs excellent relative disparity but not metric depth. We "
               "recover scale with a head conditioned on the learned focal length, so the mapping "
               "is camera-aware. We study a free convolutional head and a camera-conditioned "
               "global-affine head; Section V reports which wins."))
    s.append(H2("D", "Self-calibration and inference without LiDAR"))
    s.append(P("The intrinsics come from a hybrid module: a per-camera latent vector, optimized "
               "over the dataset in the manner of deep self-calibration, plus a residual predicted "
               "from the image so the model can re-estimate intrinsics for a camera it has not "
               "seen. LiDAR is used only to train and to validate; at inference the network "
               "consumes a single image and outputs depth, intrinsics, and hence object distance, "
               "with no LiDAR required. Because metric scale is conditioned on the estimated focal "
               "length, the learned scale is meant to transfer to a new camera once its focal "
               "length is recovered from the image. For cameras far from the training domain, such "
               "as wide-angle dashboard cameras with strong distortion, scale can be re-anchored "
               "without LiDAR using a known camera height and the ground plane, known object sizes "
               "such as standardized license plates, logged GPS or vehicle speed, or photometric "
               "self-supervision over video."))
    s.append(H2("E", "Annotated assumptions"))
    s.append(P("<b>A1</b> DAv2 output is relative affine-invariant disparity (orientation learned). "
               "<b>A2</b> Relative-to-metric is approximately a global affine in inverse-depth per "
               "image plus a bounded local residual. <b>A3</b> Metric scale correlates with focal "
               "length, hence the conditioning. <b>A4</b> Argoverse ring cameras are near-pinhole: "
               "radial distortion is small and bounded, which also keeps un-distortion convergent. "
               "<b>A5</b> Metric depth lies in 0.1 to 300 m; estimates beyond sixty meters are "
               "reported separately."))

    # ---- IV. Implementation ----
    s.append(H1("IV", "Implementation"))
    s.append(H2("A", "Data and ground truth"))
    s.append(P("We use the Argoverse 2 Sensor dataset, downloaded selectively from the public "
               "bucket (40 training and 8 validation logs, all seven ring cameras, every third "
               "LiDAR sweep, 8.3 GB). For each frame we load the camera model, read the sweep, "
               "motion-compensate it from LiDAR time to camera time using the ego-poses, and "
               "project to obtain a sparse metric depth map (about 0.4% pixel coverage, 4 to "
               "215 m). The same machinery projects 3-D cuboids to obtain per-vehicle distance. "
               "Fig. 2 shows the resulting sparse metric ground truth."))
    s.append(_fig("viz/ring_front_center_315969904949927216_depth.png",
                  "Fig. 2.&nbsp;&nbsp;LiDAR-projected ground-truth depth (motion-compensated) on a "
                  "ring-camera image. This sparse metric supervision anchors both the depth network "
                  "and the self-calibration."))
    s.append(H2("B", "Model and training"))
    s.append(P("The backbone is Depth Anything V2 Small (24.8 M parameters). A small CNN encodes "
               "the image into a calibration feature driving a hybrid learnable-intrinsics module: "
               "a per-camera latent (self-calibration) plus an image-conditioned residual that "
               "lets the model re-estimate intrinsics for unseen cameras. Images are processed at "
               "518 by 518; intrinsics and LiDAR pixels are scaled accordingly. Training is full "
               "fp32, with gradient clipping and a non-finite-step guard. Hardware is a single "
               "NVIDIA RTX 5090 (32 GB, Blackwell) with PyTorch 2.12 and CUDA 13; a three-epoch "
               "run completes in about 30 to 45 minutes."))
    s.append(H2("C", "Engineering notes"))
    s.append(P("Five non-obvious failures were fixed: fp16 overflow on metric losses (train in "
               "fp32); the field-of-view channel arc-cosine has an infinite gradient on the "
               "optical axis (clamp away from the limits); freely-learned distortion makes "
               "un-distortion diverge (bound it); a watcher command matched its own shell; and "
               "empty optimizer groups when the backbone is frozen."))

    # ---- V. Results ----
    s.append(H1("V", "Results and Discussion"))
    s.append(P("We evaluate on 200 held-out validation frames across 8 logs the model never "
               "trained on, reporting metric depth versus LiDAR, self-calibration error versus "
               "the ground-truth intrinsics, and distance-to-vehicle error versus cuboids "
               "(Table I)."))
    s.append(_table())
    s.append(P("<i>Self-calibration works everywhere.</i> Across all variants the model recovers "
               "intrinsics to within 1.4% (fx and fy) and sub-pixel principal point on held-out "
               "cameras; v3 reaches fx 0.26% and fy 0.41%. This is the central claim: the camera "
               "parameters are recreated from the image alone."))
    s.append(P("<i>A pinpointed negative result.</i> The camera-conditioned global-affine head "
               "(v2) was appealing but hurt depth (AbsRel 0.204 to 0.368). The cause was a wrong "
               "functional-form assumption (A2): we mapped depth linearly in log-depth, but "
               "disparity is affine in inverse-depth. Correcting it (v2b) recovered most of the "
               "loss but still lost to the free conv head: a single global affine is too rigid."))
    s.append(P("<i>The real lever was the backbone.</i> Unfreezing Depth Anything V2 (v3) nearly "
               "halved AbsRel (0.204 to 0.112) and raised the accuracy at threshold 1.25 from "
               "0.71 to 0.88, while giving the best calibration. Per-range vehicle distance is "
               "given in Table II: accurate up close and degrading with range, as expected for "
               "monocular depth with sparse far LiDAR. Fig. 3 shows a qualitative example."))
    s.append(_table2())
    s.append(_figure())

    # ---- V.A Cross-dataset generalization ----
    s.append(H2("F", "Cross-dataset generalization"))
    s.append(P("A single sensor rig under-tests camera-awareness, so we built a unified benchmark "
               "layer reducing every dataset to the same contract (image, sparse LiDAR depth, "
               "ground-truth intrinsics) behind one adapter, and applied the AV2-trained v3 model "
               "to KITTI and nuScenes front cameras (fx 721 and 1266; AV2 1782) under three "
               "regimes: zero-shot; adapting only the per-camera latent on twenty frames; and "
               "additionally adapting the small depth head with the aspect prior disabled "
               "(Table III, Fig. 4). Held-out qualitative results appear in Fig. 5."))
    s.append(_table3())
    s.append(P("Three findings. <i>Zero-shot transfer fails</i>, and structurally: intrinsics are a "
               "per-camera latent plus a small image residual, so an unseen camera inherits the AV2 "
               "focal (KITTI accuracy collapses to 0.03). <i>Self-calibration transfers few-shot</i>: "
               "twenty frames recover focal length to within half a percent (fx) on both datasets. "
               "<i>Calibration and metric scale are separable</i>: latent-only adaptation leaves depth "
               "unchanged, but additionally adapting the 58.8k-parameter depth head restores it "
               "dramatically (KITTI accuracy 0.026 to 0.953, nuScenes 0.184 to 0.853), matching "
               "in-domain AV2. A residual ten-to-fourteen percent error in fy persists, as vertical "
               "focal is weakly observed by the LiDAR loss."))

    # ---- VI. Conclusion ----
    s.append(H1("VI", "Conclusion"))
    s.append(P("SelfCalibDepth shows that LiDAR is a sufficient anchor to jointly learn camera "
               "self-calibration and camera-aware metric depth from a single image, on a real "
               "driving dataset and on held-out cameras. Self-calibration is the strongest result "
               "(sub-one-percent focal error); metric depth and vehicle distance are solid and "
               "were driven primarily by fine-tuning a depth foundation model rather than by "
               "architectural cleverness, a finding made visible only because every assumption "
               "was annotated and ablated. The cross-dataset study sharpens the claim: "
               "self-calibration transfers to KITTI and nuScenes with a twenty-frame adaptation, "
               "and a small depth-head adaptation transfers the metric depth too. Limitations: "
               "zero-shot transfer fails by design (the per-camera latent dominates), the residual "
               "fy error shows vertical focal is weakly observed, distortion learning is "
               "conservative, far-range depth is weak, and Lyft L5 is implemented but unevaluated "
               "(gated data). Future work: restructure theta for genuine zero-shot calibration, a "
               "reprojection-weighted schedule to tighten fy, cross-dataset vehicle metrics, a "
               "photometric self-supervised term, and a larger backbone."))

    # ---- References ----
    s.append(Paragraph("References", st_h1))
    for r in REFS:
        s.append(Paragraph(r, st_ref))

    # ---- full-width figure page (wide cross-dataset figures) ----
    from reportlab.platypus import PageBreak
    s.append(NextPageTemplate("wide"))
    s.append(PageBreak())
    s.append(_wfig("viz_paper/results_bars.png",
                   "Fig. 4.&nbsp;&nbsp;Cross-dataset generalization. AbsRel, accuracy at threshold "
                   "1.25, and focal-length error (log axis) across the three adaptation regimes for "
                   "KITTI and nuScenes; the dashed line is the in-domain Argoverse 2 result."))
    s.append(_wfig("viz_paper/cross_dataset_qualitative.png",
                   "Fig. 5.&nbsp;&nbsp;Qualitative cross-dataset results. One row per benchmark "
                   "(Argoverse 2, KITTI, nuScenes): input, ground-truth LiDAR depth, predicted "
                   "metric depth, and absolute error at the LiDAR points, on a shared scale. Row "
                   "labels give native resolution, focal length, and held-out metrics."))

    doc.build(s)
    print("wrote PAPER.pdf")


def _table():
    data = [["Ver.", "AbsRel", "RMSE", "acc.1.25", "fx err", "fy err", "Veh.<=60"],
            ["v1", "0.204", "10.14", "0.714", "0.16%", "1.37%", "--"],
            ["v2", "0.368", "11.95", "0.560", "0.55%", "0.61%", "8.47 m"],
            ["v2b", "0.249", "10.95", "0.685", "0.30%", "1.33%", "9.14 m"],
            ["v3", "0.112", "7.30", "0.884", "0.26%", "0.41%", "8.56 m"]]
    t = Table(data, colWidths=[0.36, 0.5, 0.5, 0.6, 0.48, 0.48, 0.62])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"), ("FONTNAME", (0, 4), (-1, 4), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    cap = Paragraph("TABLE I.&nbsp;&nbsp;A<font size=6>BLATION ON HELD-OUT VALIDATION</font> "
                    "(v1 free conv/frozen; v2 affine log-depth; v2b affine inverse-depth; "
                    "v3 free conv + unfrozen backbone).", st_tcap)
    from reportlab.platypus import KeepTogether
    return KeepTogether([cap, t, Spacer(1, 6)])


def _fig(path, caption):
    """Generic in-column figure: image scaled to column width + caption."""
    from reportlab.platypus import KeepTogether
    items = []
    p = Path(path)
    if p.exists():
        from PIL import Image as PILImage
        iw, ih = PILImage.open(p).size
        items.append(Image(str(p), width=COL_W, height=COL_W * ih / iw))
    items.append(Paragraph(caption, st_cap))
    items.append(Spacer(1, 5))
    return KeepTogether(items)


def _wfig(path, caption):
    """Full-page-width figure (spans both columns) + caption."""
    from reportlab.platypus import KeepTogether
    W = PAGE_W - LM - RM
    items = []
    p = Path(path)
    if p.exists():
        from PIL import Image as PILImage
        iw, ih = PILImage.open(p).size
        items.append(Image(str(p), width=W, height=W * ih / iw))
    items.append(Paragraph(caption, st_cap))
    items.append(Spacer(1, 10))
    return KeepTogether(items)


def _table3():
    from reportlab.platypus import KeepTogether
    data = [["Dataset", "Regime", "AbsRel", "acc.1.25", "fx err"],
            ["KITTI", "zero-shot", "0.390", "0.026", "97.1%"],
            ["KITTI", "+latent", "0.402", "0.021", "0.2%"],
            ["KITTI", "+latent+head", "0.100", "0.953", "1.1%"],
            ["nuSc.", "zero-shot", "0.274", "0.184", "45.3%"],
            ["nuSc.", "+latent", "0.284", "0.172", "0.5%"],
            ["nuSc.", "+latent+head", "0.122", "0.853", "0.7%"],
            ["AV2", "in-domain", "0.112", "0.884", "0.26%"]]
    t = Table(data, colWidths=[0.5, 0.92, 0.5, 0.6, 0.5])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 3), (-1, 3), "Times-Bold"), ("FONTNAME", (0, 6), (-1, 6), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, 3), (-1, 3), 0.3, colors.grey),
        ("LINEBELOW", (0, 6), (-1, 6), 0.3, colors.grey),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    cap = Paragraph("TABLE III.&nbsp;&nbsp;C<font size=6>ROSS-DATASET TRANSFER</font> "
                    "(AV2-trained v3; 20-frame few-shot; held-out frames).", st_tcap)
    return KeepTogether([cap, t, Spacer(1, 6)])


def _figure():
    return _fig("viz_v3/infer_v3_closetraffic.png",
                "Fig. 3.&nbsp;&nbsp;Single-image inference (v3). Left: input with per-vehicle "
                "predicted/ground-truth distance in meters. Right: predicted metric depth. Focal "
                "length self-estimated to within 0.4%.")


def _table2():
    from reportlab.platypus import KeepTogether
    data = [["Range", "0-30 m", "30-60 m", "60 m+", "overall"],
            ["MAE", "3.90 m", "12.57 m", "35.85 m", "22.46 m"],
            ["n", "330", "384", "741", "1455"]]
    t = Table(data, colWidths=[0.5, 0.62, 0.66, 0.6, 0.66])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"), ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"), ("FONTNAME", (0, 0), (0, -1), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    cap = Paragraph("TABLE II.&nbsp;&nbsp;V<font size=6>EHICLE-DISTANCE ERROR BY RANGE</font> (v3).",
                    st_tcap)
    return KeepTogether([cap, t, Spacer(1, 6)])


REFS = [
    "[1] D. Eigen, C. Puhrsch, and R. Fergus, &ldquo;Depth map prediction from a single image "
    "using a multi-scale deep network,&rdquo; in <i>NeurIPS</i>, 2014.",
    "[2] R. Ranftl et al., &ldquo;Towards robust monocular depth estimation,&rdquo; <i>IEEE "
    "TPAMI</i>, 2022.",
    "[3] R. Ranftl, A. Bochkovskiy, and V. Koltun, &ldquo;Vision transformers for dense "
    "prediction,&rdquo; in <i>ICCV</i>, 2021.",
    "[4] L. Yang et al., &ldquo;Depth Anything V2,&rdquo; in <i>NeurIPS</i>, 2024.",
    "[5] J. M. Facil et al., &ldquo;CAM-Convs: Camera-aware multi-scale convolutions for "
    "single-view depth,&rdquo; in <i>CVPR</i>, 2019.",
    "[6] W. Yin et al., &ldquo;Metric3D: Towards zero-shot metric 3D prediction,&rdquo; in "
    "<i>ICCV</i>, 2023.",
    "[7] S. F. Bhat et al., &ldquo;ZoeDepth: Zero-shot transfer by combining relative and metric "
    "depth,&rdquo; arXiv:2302.12288, 2023.",
    "[8] P. Hagemann et al., &ldquo;Deep geometry-aware camera self-calibration from video,&rdquo; "
    "in <i>ICCV</i>, 2023.",
    "[9] B. Wilson et al., &ldquo;Argoverse 2: Next generation datasets for self-driving "
    "perception and forecasting,&rdquo; in <i>NeurIPS Datasets and Benchmarks</i>, 2021.",
    "[10] A. Geiger, P. Lenz, and R. Urtasun, &ldquo;Are we ready for autonomous driving? The "
    "KITTI vision benchmark suite,&rdquo; in <i>CVPR</i>, 2012.",
    "[11] H. Caesar et al., &ldquo;nuScenes: A multimodal dataset for autonomous driving,&rdquo; "
    "in <i>CVPR</i>, 2020.",
]

if __name__ == "__main__":
    build()
