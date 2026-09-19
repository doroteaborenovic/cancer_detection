import os
import time
import datetime
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms

# ==============================================================================
# 1. PUNA PODRŠKA ZA PDF I SRPSKA SLOVA (Č, Ć, Š, Ž, Đ)
# ==============================================================================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Koristimo originalni Windows Arial font koji uvek podržava č, ć, š, ž, đ
    win_font = "C:/Windows/Fonts/arial.ttf"
    win_font_bold = "C:/Windows/Fonts/arialbd.ttf"
    if os.path.exists(win_font):
        pdfmetrics.registerFont(TTFont("ArialSRB", win_font))
        pdfmetrics.registerFont(TTFont("ArialSRBBold", win_font_bold if os.path.exists(win_font_bold) else win_font))
        FONT_NAME = "ArialSRB"
        FONT_BOLD = "ArialSRBBold"
    else:
        FONT_NAME = "Helvetica"
        FONT_BOLD = "Helvetica-Bold"
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    FONT_NAME = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

# ==============================================================================
# 2. MODEL I KLINIČKI MODALITETI (MRI, CT, HISTOLOGIJA)
# ==============================================================================
IMG_SIZE = (128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'all_benign', 'all_early', 'all_pre', 'all_pro', 
    'brain_glioma', 'brain_menin', 'brain_tumor', 
    'breast_benign', 'breast_malignant', 
    'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi', 
    'colon_aca', 'colon_bnt', 
    'kidney_normal', 'kidney_tumor', 
    'lung_aca', 'lung_bnt', 'lung_scc', 
    'lymph_cll', 'lymph_fl', 'lymph_mcl', 
    'oral_normal', 'oral_scc'
]

CLASS_METADATA = {
    'all_benign': ('Zdravo / Benigno', 'Krvne ćelije bez leukemije', 'Redovna kontrola krvne slike.', '🩸 Hematološki razmaz krvi'),
    'all_early': ('Maligno', 'Akutna limfoblastna leukemija (Rani stadijum)', 'Hitna citometrijska obrada i biopsija koštane srži.', '🩸 Hematološki razmaz krvi'),
    'all_pre': ('Maligno', 'Akutna limfoblastna leukemija (Pre-B faza)', 'Onkohematološki konzilijum i protokolarno lečenje.', '🩸 Hematološki razmaz krvi'),
    'all_pro': ('Maligno', 'Akutna limfoblastna leukemija (Pro-B faza)', 'Hitna hospitalizacija i sistemska terapija.', '🩸 Hematološki razmaz krvi'),
    'brain_glioma': ('Maligno', 'Gliom — infiltrativni tumor mozga', 'Hitna magnetna rezonanca (MRI sa kontrastom) i neurohirurški pregled.', '🧠 Magnetna Rezonanca (MRI Mozga)'),
    'brain_menin': ('Benigno / Granično', 'Meningiom moždanih ovojnica', 'Redovno praćenje dimenzija tumora na 6 meseci.', '🧠 Magnetna Rezonanca (MRI Mozga)'),
    'brain_tumor': ('Maligno', 'Tumor regije hipofize', 'Endokrinološki hormonski status i neurohirurška evaluacija.', '🧠 Magnetna Rezonanca (MRI Mozga)'),
    'breast_benign': ('Zdravo / Benigno', 'Benigna promena tkiva dojke (fibroadenom/cista)', 'Ultrazvučna kontrola za 6 do 12 meseci.', '🔬 Histopatologija tkiva dojke (H&E)'),
    'breast_malignant': ('Maligno', 'Karcinom dojke sa invazijom', 'Biopsija (Core-biopsy), mamografija i onkološki konzilijum.', '🔬 Histopatologija tkiva dojke (H&E)'),
    'cervix_dyk': ('Abnormalno', 'Diskeratoza pločastog epitela grlića materice', 'Kolposkopija i ponovni citološki bris.', '🔬 Citološki bris grlića (PAPA)'),
    'cervix_koc': ('Virusna lezija', 'Koilocitoza (promene izazvane HPV-om)', 'HPV tipizacija visokog rizika i kolposkopski pregled.', '🔬 Citološki bris grlića (PAPA)'),
    'cervix_mep': ('Prekancerozno', 'Metaplazija epitela grlića materice', 'Ciljana biopsija i praćenje promena na epitelu.', '🔬 Citološki bris grlića (PAPA)'),
    'cervix_pab': ('Abnormalno', 'Atipične parabazalne ćelije', 'Patohistološki pregled endocervikalnog kanala.', '🔬 Citološki bris grlića (PAPA)'),
    'cervix_sfi': ('Zdravo / Benigno', 'Normalne ćelije pločastog epitela', 'Uredan nalaz. Rutinski godišnji skrining.', '🔬 Citološki bris grlića (PAPA)'),
    'colon_aca': ('Maligno', 'Adenokarcinom debelog creva', 'Hitna kolonoskopija sa uzimanjem isečka za patologiju.', '🔬 Histopatološki presek debelog creva'),
    'colon_bnt': ('Zdravo / Benigno', 'Zdravo tkivo sluzokože debelog creva', 'Uredan nalaz bez patoloških promena.', '🔬 Histopatološki presek debelog creva'),
    'kidney_normal': ('Zdravo / Benigno', 'Normalan parenhim bubrega bez lezija', 'Nisu potrebne dalje intervencije.', '🩻 Kompjuterizovana Tomografija (CT Bubrega)'),
    'kidney_tumor': ('Maligno', 'Renalni karcinom bubrežnog tkiva', 'CT urografija sa kontrastom i pregled urologa.', '🩻 Kompjuterizovana Tomografija (CT Bubrega)'),
    'lung_aca': ('Maligno', 'Adenokarcinom pluća', 'Bronhoskopija, CT grudnog koša i onkološki konzilijum.', '🔬 Histopatologija plućnog tkiva'),
    'lung_bnt': ('Zdravo / Benigno', 'Zdravo plućno tkivo', 'Očuvana alveolarna struktura, nema maligniteta.', '🔬 Histopatologija plućnog tkiva'),
    'lung_scc': ('Maligno', 'Planocelularni karcinom pluća', 'Histološka potvrda i bronhoskopska biopsija.', '🔬 Histopatologija plućnog tkiva'),
    'lymph_cll': ('Maligno', 'Hronična limfocitna leukemija', 'Hematološki protokoli i praćenje limfnih čvorova.', '🔬 Histologija limfnih čvorova'),
    'lymph_fl': ('Maligno', 'Folikularni limfom', 'Biopsija limfnog čvora i imunohistohemija.', '🔬 Histologija limfnih čvorova'),
    'lymph_mcl': ('Maligno', 'Mantle cell limfom', 'Imunoterapijski i onkološki pristup lečenju.', '🔬 Histologija limfnih čvorova'),
    'oral_normal': ('Zdravo / Benigno', 'Zdrava sluzokoža usne duplje', 'Uredan stomatološki/maksilofacijalni nalaz.', '🔬 Histopatologija oralnog tkiva'),
    'oral_scc': ('Maligno', 'Planocelularni karcinom usne duplje', 'Hitna inciziona biopsija i pregled maksilofacijalnog hirurga.', '🔬 Histopatologija oralnog tkiva')
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )
    def forward(self, x):
        return self.block(x)

class ScratchTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout_rate=0.2),
            ConvBlock(32, 64, dropout_rate=0.25),
            ConvBlock(64, 128, dropout_rate=0.3),
            ConvBlock(128, 256, dropout_rate=0.35)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

MODEL_PATH = "best_scratch_cancer_model.pth"
model = ScratchTumorCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==============================================================================
# 3. GRAD-CAM I METRIKA LEZIJE
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE[0], IMG_SIZE[1]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

grad_cam = GradCAM(model, model.features[3].block[3])

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def draw_hud_target(raw_np, cam, alpha):
    raw_resized = cv2.resize(raw_np, (IMG_SIZE[0], IMG_SIZE[1]))
    raw_heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    raw_heat = cv2.cvtColor(raw_heat, cv2.COLOR_BGR2RGB)
    
    blended = np.uint8((1.0 - alpha) * raw_resized + alpha * raw_heat)

    # Proračun zahvaćenosti lezije
    threshold = 0.55
    hotspot_mask = np.uint8(cam > threshold) * 255
    total_pixels = IMG_SIZE[0] * IMG_SIZE[1]
    lesion_pixels = np.sum(cam > threshold)
    area_percent = (lesion_pixels / total_pixels) * 100

    contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hud_overlay = blended.copy()
    box_coords = "Žarište u granicama normale"
    
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) > 15:
            x, y, w, h = cv2.boundingRect(largest_c)
            box_coords = f"X:{x} Y:{y} [Širina:{w}px Visina:{h}px]"
            
            # Crtanje sajber HUD nišana (Neon Cyan)
            color = (0, 242, 254)
            corner_len = max(6, int(min(w, h) * 0.25))
            
            # 4 ugla oko lezije
            cv2.line(hud_overlay, (x, y), (x + corner_len, y), color, 2)
            cv2.line(hud_overlay, (x, y), (x, y + corner_len), color, 2)
            cv2.line(hud_overlay, (x + w, y), (x + w - corner_len, y), color, 2)
            cv2.line(hud_overlay, (x + w, y), (x + w, y + corner_len), color, 2)
            cv2.line(hud_overlay, (x, y + h), (x + corner_len, y + h), color, 2)
            cv2.line(hud_overlay, (x, y + h), (x, y + h - corner_len), color, 2)
            cv2.line(hud_overlay, (x + w, y + h), (x + w - corner_len, y + h), color, 2)
            cv2.line(hud_overlay, (x + w, y + h), (x + w, y + h - corner_len), color, 2)
            
            # Centralni nišan lekara
            cx, cy = x + w // 2, y + h // 2
            cv2.drawMarker(hud_overlay, (cx, cy), (255, 50, 100), cv2.MARKER_CROSS, 8, 1)

    return hud_overlay, raw_heat, area_percent, box_coords

# ==============================================================================
# 4. GENERISANJE PDF-A SA PUNIM Č, Ć, Š, Ž, Đ
# ==============================================================================
def create_pdf_report(raw_np, raw_heat, result_data):
    if not HAS_REPORTLAB or result_data is None:
        return None
    
    top_class = result_data.get('class', 'N/A')
    confidence = result_data.get('conf', 0.0)
    status_type = result_data.get('status', 'N/A')
    desc = result_data.get('desc', 'N/A')
    recom = result_data.get('recom', 'N/A')
    modality = result_data.get('modality', 'N/A')
    area_pct = result_data.get('area', 0.0)

    temp_orig = "temp_orig.jpg"
    temp_heat = "temp_heat.jpg"
    Image.fromarray(cv2.resize(raw_np, (180, 180))).save(temp_orig)
    Image.fromarray(cv2.resize(raw_heat, (180, 180))).save(temp_heat)

    pdf_filename = f"Medicinski_Nalaz_{top_class}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', fontName=FONT_BOLD, fontSize=18, textColor=colors.HexColor("#0f172a"), spaceAfter=8)
    normal_style = ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=10, textColor=colors.HexColor("#334155"), leading=14)
    bold_style = ParagraphStyle('Bold', fontName=FONT_BOLD, fontSize=10, textColor=colors.HexColor("#0f172a"))

    story = []
    story.append(Paragraph("KLINIČKI NALAZ SISTEMA ZA DETEKCIJU TUMORA", title_style))
    story.append(Paragraph(f"<b>Datum:</b> {datetime.datetime.now().strftime('%d.%m.%Y. u %H:%M')} | <b>Modalitet:</b> {modality}", normal_style))
    story.append(Spacer(1, 14))

    img1 = RLImage(temp_orig, width=170, height=170)
    img2 = RLImage(temp_heat, width=170, height=170)
    img_table = Table([[img1, img2], [Paragraph("<b>1. Ulazni medicinski snimak</b>", normal_style), Paragraph("<b>2. Grad-CAM lokalizacija žarišta</b>", normal_style)]], colWidths=[240, 240])
    img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(img_table)
    story.append(Spacer(1, 14))

    status_color = colors.HexColor("#ef4444") if "Maligno" in status_type else colors.HexColor("#10b981")
    diag_data = [
        [Paragraph("<b>Identifikovana klasa:</b>", normal_style), Paragraph(f"<b>{top_class}</b>", bold_style)],
        [Paragraph("<b>Status tkiva:</b>", normal_style), Paragraph(f"<b>{status_type.upper()}</b>", ParagraphStyle('Status', fontName=FONT_BOLD, textColor=status_color))],
        [Paragraph("<b>Pouzdanost modela:</b>", normal_style), Paragraph(f"<b>{confidence:.2f}%</b>", bold_style)],
        [Paragraph("<b>Zahvaćenost vidnog polja:</b>", normal_style), Paragraph(f"<b>{area_pct:.1f}%</b> površine lezije", bold_style)],
        [Paragraph("<b>Patološki opis:</b>", normal_style), Paragraph(desc, normal_style)],
        [Paragraph("<b>Preporučeni sledeći korak:</b>", normal_style), Paragraph(recom, normal_style)]
    ]
    diag_table = Table(diag_data, colWidths=[150, 330])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<i>*Napomena: Sistem koristi duboke konvolucione mreže i XAI toplotne mape (97.59% tačnost na validaciji). Rezultat je stručna pomoćna dijagnostika i zahteva overu lekara specijaliste.</i>", ParagraphStyle('Disc', fontName=FONT_NAME, fontSize=8, textColor=colors.HexColor("#64748b"))))

    doc.build(story)
    return pdf_filename

# ==============================================================================
# 5. LIVE VIDEO / RADAR SKENIRANJE
# ==============================================================================
def live_hud_scan(image, alpha):
    if image is None:
        yield None, None, {}, "<div style='color: #64748b; text-align: center;'>Otpremite snimak za analizu.</div>", None, None, None
        return

    raw_original = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    raw_np = np.array(raw_original)
    img_tensor = transform(raw_original).unsqueeze(0).to(device)

    # Animacija radarskog laserskog skenera (Live video efekat)
    h, w, _ = raw_np.shape
    for scan_y in range(15, h, 28):
        scan_frame = raw_np.copy()
        cv2.line(scan_frame, (0, scan_y), (w, scan_y), (0, 242, 254), 2)
        cv2.line(scan_frame, (0, max(0, scan_y - 4)), (w, max(0, scan_y - 4)), (0, 160, 220), 1)
        scan_status = f"""
        <div style='background: rgba(0, 242, 254, 0.08); border-left: 4px solid #00f2fe; color: #38bdf8; padding: 10px 16px; border-radius: 8px; font-family: monospace;'>
            🛰️ [RADAR SCAN] Optičko skeniranje snimka na Y:{scan_y}px... Segmentacija tkiva...
        </div>
        """
        yield None, scan_frame, {}, scan_status, None, None, None
        time.sleep(0.08)

    # Predikcija
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    top5_probs, top5_indices = torch.topk(probs, 5)
    top_pred_idx = top5_indices[0].item()
    top_pred_class = class_names[top_pred_idx]
    top_confidence = top5_probs[0].item() * 100
    confidences = {class_names[idx.item()]: float(prob.item()) for prob, idx in zip(top5_probs, top5_indices)}

    # Generisanje Grad-CAM i HUD nišana
    cam = grad_cam.generate(img_tensor, target_class=top_pred_idx)
    hud_overlay, raw_heat, area_pct, box_coords = draw_hud_target(raw_np, cam, alpha)

    status_type, description, next_steps, modality = CLASS_METADATA.get(
        top_pred_class, ('Nepoznato', 'Opis nije dostupan', 'Konsultovati lekara.', 'Medicinski snimak')
    )
    is_malignant = "Maligno" in status_type or "Abnormalno" in status_type or "Virus" in status_type

    badge_bg = "rgba(239, 68, 68, 0.15)" if is_malignant else "rgba(16, 185, 129, 0.15)"
    badge_border = "#ef4444" if is_malignant else "#10b981"
    badge_color = "#f87171" if is_malignant else "#34d399"
    badge_text = "MALIGNI STATUS DETEKTOVAN" if is_malignant else "BENIGNO / NORMALNO STANJE"

    report_html = f"""
    <div style="background: rgba(10, 15, 29, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px; margin-top: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 12px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 13px; font-weight: 800; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase;">KLINIČKI NALAZ</span>
                <span style="background: rgba(56, 189, 248, 0.12); border: 1px solid #38bdf8; color: #38bdf8; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">{modality}</span>
            </div>
            <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 800;">
                {badge_text}
            </span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px;">
            <div style="background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; border-left: 3px solid #38bdf8;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Detektovana klasa</div>
                <div style="font-size: 17px; font-weight: 800; color: #f8fafc; font-family: monospace; margin-top: 3px;">{top_pred_class}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; border-left: 3px solid #fbbf24;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Pouzdanost (Confidence)</div>
                <div style="font-size: 17px; font-weight: 800; color: #fbbf24; margin-top: 3px;">{top_confidence:.2f}%</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; border-left: 3px solid #a855f7;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Zahvaćenost tkiva</div>
                <div style="font-size: 17px; font-weight: 800; color: #c084fc; margin-top: 3px;">{area_pct:.1f}% polja</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; border-left: 3px solid #06b6d4;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Koordinate žarišta</div>
                <div style="font-size: 12px; font-weight: 700; color: #22d3ee; font-family: monospace; margin-top: 5px;">{box_coords}</div>
            </div>
        </div>

        <div style="background: rgba(0, 0, 0, 0.3); padding: 14px; border-radius: 10px; margin-bottom: 10px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 13px; margin-bottom: 4px;">📖 Patohistološko tumačenje:</div>
            <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5;">{description}</div>
        </div>
        <div style="background: rgba(0, 0, 0, 0.3); padding: 14px; border-radius: 10px;">
            <div style="color: #a855f7; font-weight: 700; font-size: 13px; margin-bottom: 4px;">🩺 Preporučeni klinički korak:</div>
            <div style="color: #e2e8f0; font-size: 13px; line-height: 1.5;">{next_steps}</div>
        </div>
    </div>
    """

    payload = {
        'class': top_pred_class, 'conf': top_confidence, 'status': status_type,
        'desc': description, 'recom': next_steps, 'modality': modality, 'area': area_pct
    }

    yield raw_heat, hud_overlay, confidences, report_html, raw_np, cam, payload

def slider_change(raw_np, cam, alpha):
    if raw_np is None or cam is None:
        return None
    hud_overlay, _, _, _ = draw_hud_target(raw_np, cam, alpha)
    return hud_overlay

# ==============================================================================
# 6. FENSI CYBER-DARK MINIMALIST UI
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-color: #030712 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.title-container {
    text-align: center;
    padding: 26px 10px 16px 10px;
    margin-bottom: 10px;
}
.glow-title {
    font-size: 38px;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #9d4edd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-transform: uppercase;
}
.subtitle-text {
    color: #64748b;
    font-size: 15px;
    margin-top: 8px;
    font-weight: 500;
}
button.primary {
    background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%) !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    font-weight: 700 !important;
}
.pdf-btn {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    color: white !important;
}
"""

with gr.Blocks(css=custom_css, title="Detekcija Tumora") as demo:
    state_raw = gr.State(None)
    state_cam = gr.State(None)
    state_data = gr.State(None)

    gr.HTML("""
        <div class="title-container">
            <h1 class="glow-title">🔬 DETEKCIJA TUMORA</h1>
            <p class="subtitle-text">Neuronska mreža za detekciju malignih ćelija i vizuelnu analizu patoloških promena</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="1. Ulazni snimak (MRI / CT / Histologija)", type="numpy")
            btn = gr.Button("⚡ Pokreni Live Skeniranje & Analizu", variant="primary", size="lg")
            
            slider_opacity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                label="🎚️ Providnost toplotne mape (Grad-CAM Blend)"
            )

        with gr.Column(scale=1):
            heatmap_view = gr.Image(label="2. Sirova toplotna mapa pažnje", interactive=False)

        with gr.Column(scale=1):
            overlay_view = gr.Image(label="3. HUD Nišan lezije & Fokus tkiva", interactive=False)

    # Brzi primeri iz sample_images foldera
    sample_dir = "sample_images"
    example_files = []
    if os.path.exists(sample_dir):
        for f in sorted(os.listdir(sample_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                example_files.append([os.path.join(sample_dir, f)])

    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=input_img,
            label="💡 Brzi primeri sa klastera (MRI Mozak, CT Bubreg, Pluća, Dojka...)"
        )

    with gr.Row():
        with gr.Column(scale=1):
            prob_bars = gr.Label(num_top_classes=5, label="Distribucija verovatnoća (Top 5)")
            btn_pdf = gr.Button("📄 Preuzmi Zvanični PDF Nalaz", elem_classes=["pdf-btn"], size="sm")
            pdf_file_output = gr.File(label="Preuzimanje PDF nalaza", interactive=False)

        with gr.Column(scale=2):
            report_display = gr.HTML("<div style='color: #475569; text-align: center; padding: 30px;'>Otpremite snimak i kliknite na 'Pokreni Live Skeniranje'.</div>")

    # Povezivanje događaja
    btn.click(
        fn=live_hud_scan,
        inputs=[input_img, slider_opacity],
        outputs=[heatmap_view, overlay_view, prob_bars, report_display, state_raw, state_cam, state_data]
    )

    slider_opacity.change(
        fn=slider_change,
        inputs=[state_raw, state_cam, slider_opacity],
        outputs=overlay_view
    )

    btn_pdf.click(
        fn=create_pdf_report,
        inputs=[state_raw, heatmap_view, state_data],
        outputs=pdf_file_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)