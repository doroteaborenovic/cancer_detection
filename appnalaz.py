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

# Za generisanje PDF-a
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ==============================================================================
# 1. KONFIGURACIJA I MODEL
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

CLASS_INFO = {
    'all_benign': ('Zdravo / Benigno', 'Krvne ćelije bez leukemije', 'Redovna kontrola krvne slike.'),
    'all_early': ('Maligno', 'Akutna limfoblastna leukemija (Rani stadijum)', 'Hitna hematološka obrada i biopsija koštane srži.'),
    'all_pre': ('Maligno', 'Akutna limfoblastna leukemija (Pre-B faza)', 'Onkohematološki konzilijum i protokolarno lečenje.'),
    'all_pro': ('Maligno', 'Akutna limfoblastna leukemija (Pro-B faza)', 'Hitna hospitalizacija i sistemska hemioterapija.'),
    'brain_glioma': ('Maligno', 'Gliom — infiltrativni tumor mozga', 'Hitna magnetna rezonanca (MRI sa kontrastom) i pregled neurohirurga.'),
    'brain_menin': ('Benigno / Granično', 'Meningiom moždanih ovojnica', 'Redovno praćenje dimenzija tumora na 6 meseci.'),
    'brain_tumor': ('Maligno', 'Tumor regije hipofize', 'Endokrinološki hormonski status i neurohirurška evaluacija.'),
    'breast_benign': ('Zdravo / Benigno', 'Benigna promena tkiva dojke', 'Ultrazvučna kontrola na 6 do 12 meseci.'),
    'breast_malignant': ('Maligno', 'Karcinom dojke', 'Biopsija (Core-biopsy), mamografija i onkološki konzilijum.'),
    'cervix_dyk': ('Abnormalno', 'Diskeratoza epitela grlića materice', 'Kolposkopija i ponovni citološki PAPA bris.'),
    'cervix_koc': ('Virusna lezija', 'Koilocitoza (HPV promena)', 'HPV tipizacija visokog rizika i kolposkopski pregled.'),
    'cervix_mep': ('Prekancerozno', 'Metaplazija ćelija grlića materice', 'Ciljana biopsija i praćenje promena na epitelu.'),
    'cervix_pab': ('Abnormalno', 'Atipične parabazalne ćelije', 'Patohistološki pregled endocervikalnog kanala.'),
    'cervix_sfi': ('Zdravo / Benigno', 'Normalne ćelije grlića materice', 'Uredan nalaz. Rutinski godišnji skrining.'),
    'colon_aca': ('Maligno', 'Adenokarcinom debelog creva', 'Hitna kolonoskopija sa uzimanjem uzorka za patologiju.'),
    'colon_bnt': ('Zdravo / Benigno', 'Zdravo tkivo debelog creva', 'Uredan nalaz bez patoloških alteracija.'),
    'kidney_normal': ('Zdravo / Benigno', 'Normalno tkivo bubrega', 'Uredan nalaz parenhima bubrega.'),
    'kidney_tumor': ('Maligno', 'Tumor bubrega (Renalni karcinom)', 'CT urografija sa kontrastom i pregled urologa.'),
    'lung_aca': ('Maligno', 'Adenokarcinom pluća', 'Bronhoskopija, CT grudnog koša i onkološki konzilijum.'),
    'lung_bnt': ('Zdravo / Benigno', 'Zdravo plućno tkivo', 'Očuvana alveolarna struktura, nema maligniteta.'),
    'lung_scc': ('Maligno', 'Planocelularni karcinom pluća', 'Histološka potvrda i bronhoskopska biopsija.'),
    'lymph_cll': ('Maligno', 'Hronična limfocitna leukemija', 'Hematološki protokoli i praćenje limfnih čvorova.'),
    'lymph_fl': ('Maligno', 'Folikularni limfom', 'Biopsija limfnog čvora i imunohistohemija.'),
    'lymph_mcl': ('Maligno', 'Mantle cell limfom', 'Imunoterapijski i onkološki pristup lečenju.'),
    'oral_normal': ('Zdravo / Benigno', 'Zdrava sluzokoža usne duplje', 'Uredan stomatološki/maksilofacijalni nalaz.'),
    'oral_scc': ('Maligno', 'Planocelularni karcinom usne duplje', 'Hitna inciziona biopsija i pregled maksilofacijalnog hirurga.')
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
# 2. GRAD-CAM
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

# ==============================================================================
# 3. INTERAKTIVNO MEŠANJE TOKOM PROMENE SLAJDERA
# ==============================================================================
def update_opacity(raw_np, raw_heat, alpha):
    if raw_np is None or raw_heat is None:
        return None
    raw_resized = cv2.resize(raw_np, (IMG_SIZE[0], IMG_SIZE[1]))
    blended = np.uint8((1.0 - alpha) * raw_resized + alpha * raw_heat)
    return blended

# ==============================================================================
# 4. GENERISANJE PDF IZVEŠTAJA
# ==============================================================================
def create_pdf_report(raw_np, raw_heat, result_data):
    if not HAS_REPORTLAB or result_data is None:
        return None
    
    top_class = result_data.get('class', 'N/A')
    confidence = result_data.get('conf', 0.0)
    status_type = result_data.get('status', 'N/A')
    desc = result_data.get('desc', 'N/A')
    recom = result_data.get('recom', 'N/A')

    # Snimi privremene slike za PDF
    temp_orig_path = "temp_orig.jpg"
    temp_heat_path = "temp_heat.jpg"
    Image.fromarray(cv2.resize(raw_np, (200, 200))).save(temp_orig_path)
    Image.fromarray(cv2.resize(raw_heat, (200, 200))).save(temp_heat_path)

    pdf_filename = f"Medicinski_Nalaz_{top_class}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor("#0f172a"), spaceAfter=10)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#334155"), leading=14)
    bold_style = ParagraphStyle('Bold', parent=styles['Normal'], fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#0f172a"))

    story = []
    story.append(Paragraph("🔬 KLINIČKI NALAZ SISTEMA ZA DETEKCIJU TUMORA", title_style))
    story.append(Paragraph(f"<b>Datum izveštaja:</b> {datetime.datetime.now().strftime('%d.%m.%Y. u %H:%M:%S')} | <b>Metoda:</b> Deep CNN + Grad-CAM Explainable AI", normal_style))
    story.append(Spacer(1, 15))

    # Tabela sa slikama
    img1 = RLImage(temp_orig_path, width=180, height=180)
    img2 = RLImage(temp_heat_path, width=180, height=180)
    img_table = Table([[img1, img2], [Paragraph("<b>1. Ulazni histopatološki snimak</b>", normal_style), Paragraph("<b>2. Grad-CAM mapa pažnje (Žarište)</b>", normal_style)]], colWidths=[240, 240])
    img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    # Status i dijagnoza
    status_color = colors.HexColor("#ef4444") if "Maligno" in status_type else colors.HexColor("#10b981")
    diag_data = [
        [Paragraph("<b>Identifikovana klasa:</b>", normal_style), Paragraph(f"<b>{top_class}</b>", bold_style)],
        [Paragraph("<b>Status tkiva:</b>", normal_style), Paragraph(f"<b>{status_type.upper()}</b>", ParagraphStyle('Status', parent=bold_style, textColor=status_color))],
        [Paragraph("<b>Pouzdanost modela:</b>", normal_style), Paragraph(f"<b>{confidence:.2f}%</b>", bold_style)],
        [Paragraph("<b>Patološki opis:</b>", normal_style), Paragraph(desc, normal_style)],
        [Paragraph("<b>Preporučeni korak:</b>", normal_style), Paragraph(recom, normal_style)]
    ]
    diag_table = Table(diag_data, colWidths=[140, 340])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<i>*Napomena: Ovaj nalaz generiše konvolucioni model veštačke inteligencije (95.96% tačnost na test skupu). Služi kao softverska podrška lekaru specijalisti i zahteva konačnu histopatološku verifikaciju.</i>", ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, textColor=colors.HexColor("#64748b"))))

    doc.build(story)
    return pdf_filename

# ==============================================================================
# 5. GLAVNI LIVE PIPELINE
# ==============================================================================
def run_pipeline(image, alpha):
    if image is None:
        yield None, None, {}, "<div style='color: #64748b; text-align: center;'>Otpremite sliku za analizu.</div>", None, None, None
        return

    img_pil = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Korak 1: Status
    status_step1 = """
    <div style='background: rgba(14, 165, 233, 0.1); border: 1px solid #0ea5e9; color: #38bdf8; padding: 12px 18px; border-radius: 10px; font-weight: 600;'>
        ⚡ Skeniranje i ekstrakcija morfoloških karakteristika tkiva...
    </div>
    """
    yield None, None, {}, status_step1, None, None, None
    time.sleep(0.35)

    # Korak 2: Predikcija
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    top5_probs, top5_indices = torch.topk(probs, 5)
    top_pred_idx = top5_indices[0].item()
    top_pred_class = class_names[top_pred_idx]
    top_confidence = top5_probs[0].item() * 100
    confidences = {class_names[idx.item()]: float(prob.item()) for prob, idx in zip(top5_probs, top5_indices)}

    cam = grad_cam.generate(img_tensor, target_class=top_pred_idx)
    raw_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    raw_heatmap = cv2.cvtColor(raw_heatmap, cv2.COLOR_BGR2RGB)

    status_step2 = """
    <div style='background: rgba(234, 179, 8, 0.1); border: 1px solid #eab308; color: #fde047; padding: 12px 18px; border-radius: 10px; font-weight: 600;'>
        🔥 Generisanje toplotne mape pažnje (Grad-CAM)...
    </div>
    """
    yield raw_heatmap, None, confidences, status_step2, None, None, None
    time.sleep(0.35)

    # Korak 3: Spajanje sa zadatom providnošću
    raw_np = np.array(img_pil)
    overlay = update_opacity(raw_np, raw_heatmap, alpha)

    status_type, description, next_steps = CLASS_INFO.get(
        top_pred_class, ('Nepoznato', 'Opis nije dostupan', 'Konsultovati lekara specijalistu.')
    )
    is_malignant = "Maligno" in status_type or "Abnormalno" in status_type or "Virus" in status_type
    
    badge_bg = "rgba(239, 68, 68, 0.15)" if is_malignant else "rgba(16, 185, 129, 0.15)"
    badge_border = "#ef4444" if is_malignant else "#10b981"
    badge_color = "#f87171" if is_malignant else "#34d399"
    badge_text = "MALIGNI STATUS DETEKTOVAN" if is_malignant else "BENIGNO / NORMALNO STANJE"

    report_html = f"""
    <div style="background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px; margin-top: 10px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 13px; font-weight: 700; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase;">DIJAGNOSTIČKI REZULTAT</span>
            <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">
                {badge_text}
            </span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 16px;">
            <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); padding: 14px; border-radius: 12px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Detektovana klasa</div>
                <div style="font-size: 18px; font-weight: 800; color: #38bdf8; font-family: monospace; margin-top: 4px;">{top_pred_class}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); padding: 14px; border-radius: 12px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Sigurnost (Confidence)</div>
                <div style="font-size: 18px; font-weight: 800; color: #fbbf24; margin-top: 4px;">{top_confidence:.2f}%</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); padding: 14px; border-radius: 12px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Medicinski opis</div>
                <div style="font-size: 14px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">{description}</div>
            </div>
        </div>
    </div>
    """

    data_payload = {
        'class': top_pred_class,
        'conf': top_confidence,
        'status': status_type,
        'desc': description,
        'recom': next_steps
    }

    yield raw_heatmap, overlay, confidences, report_html, raw_np, raw_heatmap, data_payload

# ==============================================================================
# 6. FENSI CRNI UI (CYBER-DARK)
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-color: #04060c !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.title-container {
    text-align: center;
    padding: 25px 10px 15px 10px;
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
    transition: all 0.3s ease !important;
    font-weight: 700 !important;
}
button.primary:hover {
    box-shadow: 0 6px 28px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-1px) !important;
}

.pdf-btn {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    color: white !important;
}
"""

with gr.Blocks(css=custom_css, title="Detekcija Tumora") as demo:
    # Sačuvano stanje za slajder i PDF
    state_raw = gr.State(None)
    state_heat = gr.State(None)
    state_data = gr.State(None)

    gr.HTML("""
        <div class="title-container">
            <h1 class="glow-title">🔬 DETEKCIJA TUMORA</h1>
            <p class="subtitle-text">Neuronska mreža za detekciju malignih ćelija i vizuelnu analizu patoloških promena</p>
        </div>
    """)

    # Gornji red: Ulaz, Toplotna mapa, Pomešana slika
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ulazna slika tkiva / snimak", type="numpy")
            btn = gr.Button("⚡ Pokreni Analizu", variant="primary", size="lg")
            
            # 🎚️ Interaktivni slajder za mešanje
            slider_opacity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                label="🎚️ Providnost toplotne mape (Grad-CAM Blend)"
            )

        with gr.Column(scale=1):
            heatmap_view = gr.Image(label="Toplotna mapa pažnje (Grad-CAM)", interactive=False)

        with gr.Column(scale=1):
            overlay_view = gr.Image(label="Fokus pažnje na tkivu", interactive=False)

    # 💡 Galerija primera na 1 klik (iz sample_images foldera)
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
            label="💡 Brzi primeri za testiranje (Kliknite na sliku za trenutan unos)"
        )

    # Srednji red: Distribucija i Izveštaj
    with gr.Row():
        with gr.Column(scale=1):
            prob_bars = gr.Label(num_top_classes=5, label="Verovatnoće (Top 5)")
            
            # 📄 Dugme za preuzimanje PDF-a
            btn_pdf = gr.Button("📄 Preuzmi Zvanični PDF Nalaz", elem_classes=["pdf-btn"], size="sm")
            pdf_file_output = gr.File(label="Preuzimanje PDF dokumenta", interactive=False)

        with gr.Column(scale=2):
            report_display = gr.HTML("<div style='color: #475569; text-align: center; padding: 30px;'>Otpremite sliku i kliknite na 'Pokreni Analizu'.</div>")

    # 1. Pokretanje glavnog modela
    btn.click(
        fn=run_pipeline,
        inputs=[input_img, slider_opacity],
        outputs=[heatmap_view, overlay_view, prob_bars, report_display, state_raw, state_heat, state_data]
    )

    # 2. Slajder menja providnost uživo
    slider_opacity.change(
        fn=update_opacity,
        inputs=[state_raw, state_heat, slider_opacity],
        outputs=overlay_view
    )

    # 3. Generisanje PDF-a na klik
    btn_pdf.click(
        fn=create_pdf_report,
        inputs=[state_raw, state_heat, state_data],
        outputs=pdf_file_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)