import os
import time
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms

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

# Kratak i jasan rečnik (bez suvišnog teksta)
CLASS_INFO = {
    'all_benign': ('Zdravo / Benigno', 'Krvne ćelije bez leukemije'),
    'all_early': ('Maligno', 'Akutna limfoblastna leukemija (Rani stadijum)'),
    'all_pre': ('Maligno', 'Akutna limfoblastna leukemija (Pre-B faza)'),
    'all_pro': ('Maligno', 'Akutna limfoblastna leukemija (Pro-B faza)'),
    'brain_glioma': ('Maligno', 'Gliom na mozgu'),
    'brain_menin': ('Benigno / Granično', 'Meningiom moždanih ovojnica'),
    'brain_tumor': ('Maligno', 'Tumor regije hipofize'),
    'breast_benign': ('Zdravo / Benigno', 'Benigna promena tkiva dojke'),
    'breast_malignant': ('Maligno', 'Karcinom dojke'),
    'cervix_dyk': ('Abnormalno', 'Diskeratoza epitela grlića materice'),
    'cervix_koc': ('Virusna lezija', 'Koilocitoza (HPV promena)'),
    'cervix_mep': ('Prekancerozno', 'Metaplazija ćelija grlića materice'),
    'cervix_pab': ('Abnormalno', 'Atipične parabazalne ćelije'),
    'cervix_sfi': ('Zdravo / Benigno', 'Normalne ćelije grlića materice'),
    'colon_aca': ('Maligno', 'Adenokarcinom debelog creva'),
    'colon_bnt': ('Zdravo / Benigno', 'Zdravo tkivo debelog creva'),
    'kidney_normal': ('Zdravo / Benigno', 'Normalno tkivo bubrega'),
    'kidney_tumor': ('Maligno', 'Tumor bubrega (Renalni karcinom)'),
    'lung_aca': ('Maligno', 'Adenokarcinom pluća'),
    'lung_bnt': ('Zdravo / Benigno', 'Zdravo plućno tkivo'),
    'lung_scc': ('Maligno', 'Planocelularni karcinom pluća'),
    'lymph_cll': ('Maligno', 'Hronična limfocitna leukemija'),
    'lymph_fl': ('Maligno', 'Folikularni limfom'),
    'lymph_mcl': ('Maligno', 'Mantle cell limfom'),
    'oral_normal': ('Zdravo / Benigno', 'Zdrava sluzokoža usne duplje'),
    'oral_scc': ('Maligno', 'Karcinom usne duplje')
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
# 3. LIVE PIPELINE
# ==============================================================================
def run_pipeline(image):
    if image is None:
        yield None, None, {}, "<div style='color: #64748b; text-align: center;'>Otpremite sliku za analizu.</div>"
        return

    img_pil = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Korak 1: Live indikator
    status_step1 = """
    <div style='background: rgba(14, 165, 233, 0.1); border: 1px solid #0ea5e9; color: #38bdf8; padding: 12px 18px; border-radius: 10px; font-weight: 600;'>
        ⚡ Skeniranje i ekstrakcija morfoloških karakteristika...
    </div>
    """
    yield None, None, {}, status_step1
    time.sleep(0.35)

    # Korak 2: Predikcija i Grad-CAM
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
    yield raw_heatmap, None, confidences, status_step2
    time.sleep(0.35)

    # Korak 3: Spajanje
    raw_np = np.array(img_pil)
    overlay = np.uint8(0.6 * raw_np + 0.4 * raw_heatmap)

    status_type, description = CLASS_INFO.get(top_pred_class, ('Nepoznato', 'Opis nije dostupan'))
    is_malignant = "Maligno" in status_type or "Abnormalno" in status_type or "Virus" in status_type
    
    badge_bg = "rgba(239, 68, 68, 0.15)" if is_malignant else "rgba(16, 185, 129, 0.15)"
    badge_border = "#ef4444" if is_malignant else "#10b981"
    badge_color = "#f87171" if is_malignant else "#34d399"
    badge_text = "MALIGNI STATUS" if is_malignant else "BENIGNO / NORMALNO"

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
                <div style="font-size: 17px; font-weight: 800; color: #38bdf8; font-family: monospace; margin-top: 4px;">{top_pred_class}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); padding: 14px; border-radius: 12px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Sigurnost (Confidence)</div>
                <div style="font-size: 17px; font-weight: 800; color: #fbbf24; margin-top: 4px;">{top_confidence:.2f}%</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); padding: 14px; border-radius: 12px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Medicinski opis</div>
                <div style="font-size: 14px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">{description}</div>
            </div>
        </div>
    </div>
    """

    yield raw_heatmap, overlay, confidences, report_html

# ==============================================================================
# 4. FENSI CRNI UI (CYBER-DARK MINIMALIST)
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-color: #04060c !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Minimalist neon naslov */
.title-container {
    text-align: center;
    padding: 30px 10px 20px 10px;
    margin-bottom: 20px;
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
    letter-spacing: 0.3px;
    font-weight: 500;
}

/* Fensi dugme sa glow efektom */
button.primary {
    background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%) !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.3s ease !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
button.primary:hover {
    box-shadow: 0 6px 28px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* Kartice za slike */
.image-container, .gr-box {
    background-color: #0b0f19 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
}
"""

with gr.Blocks(css=custom_css, title="Detekcija Tumora") as demo:
    # Čist minimalistički naslov
    gr.HTML("""
        <div class="title-container">
            <h1 class="glow-title">🔬 DETEKCIJA TUMORA</h1>
            <p class="subtitle-text">Neuronska mreža za detekciju malignih ćelija i vizuelnu analizu patoloških promena</p>
        </div>
    """)

    # Gornji red: Ulaz i dva prikaza pažnje
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ulazna slika tkiva / snimak", type="numpy")
            btn = gr.Button("⚡ Pokreni Analizu", variant="primary", size="lg")

        with gr.Column(scale=1):
            heatmap_view = gr.Image(label="Toplotna mapa pažnje (Grad-CAM)", interactive=False)

        with gr.Column(scale=1):
            overlay_view = gr.Image(label="Fokus pažnje na tkivu", interactive=False)

    # Srednji red: Raspodela verovatnoća i Izveštaj
    with gr.Row():
        with gr.Column(scale=1):
            prob_bars = gr.Label(num_top_classes=5, label="Verovatnoće (Top 5)")
        with gr.Column(scale=2):
            report_display = gr.HTML("<div style='color: #475569; text-align: center; padding: 30px;'>Otpremite sliku i kliknite na 'Pokreni Analizu'.</div>")

    btn.click(
        fn=run_pipeline,
        inputs=input_img,
        outputs=[heatmap_view, overlay_view, prob_bars, report_display]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)